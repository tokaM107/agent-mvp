from langchain.tools import tool
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import os
import psycopg2
import time
from models.trip_price_class import TripPricePredictor, load_model
from typing import TypedDict, List
import pandas as pd

# --- FIX: Handle the typo in the saved model file ---
import models.trip_price_class
models.trip_price_class.TripPricePreedictor = TripPricePredictor
# Ensure pickle that references __main__.TripPricePreedictor resolves during `python test_agent.py`
import sys as _sys
try:
    if '__main__' in _sys.modules:
        setattr(_sys.modules['__main__'], 'TripPricePreedictor', TripPricePredictor)
        setattr(_sys.modules['__main__'], 'TripPricePredictor', TripPricePredictor)
except Exception:
    pass
# ----------------------------------------------------

try:
    PRICE_MODEL = load_model("models/trip_price_model.joblib")
    print("✅ Price Model Loaded Successfully")
except Exception as e:
    PRICE_MODEL = None
    print(f"⚠️ Warning: Could not load price model: {e}")

class JourneyCosts(TypedDict):
    money: float
    walk: float

class Journey(TypedDict):
    path: List[str]
    costs: JourneyCosts

def get_db_connection():
    host = os.environ.get("DB_HOST", "localhost")
    db = os.environ.get("DB_NAME", "transport_db")
    user = os.environ.get("DB_USER", "postgres")
    pwd = os.environ.get("DB_PASSWORD", "postgres")
    try:
        return psycopg2.connect(host=host, database=db, user=user, password=pwd)
    except Exception:
        return None

def geocode_address(address: str) -> dict:
    geolocator = Nominatim(user_agent="alex-transport-agent")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    query = address.strip()
    if "Alexandria" not in query and "الإسكندرية" not in query:
        query = f"{query}, Alexandria, Egypt"
    try:
        location = geocode(query)
        if location:
            return {"lat": location.latitude, "lon": location.longitude}
    except:
        pass
    return {"error": "Location not found"}

def search_stop_by_name_db(name: str, limit: int = 5):
    """
    بحث دقيق لتقليل العشوائية:
    1. ILIKE: يبحث عن تطابق دقيق أو جزئي (Exact/Partial).
    2. pg_trgm: يستخدم فقط كملاذ أخير لو ملقاش حاجة.
    """
    conn = get_db_connection()
    if not conn: return []
    try:
        cur = conn.cursor()
        
        # 1. البحث الدقيق (الأولوية)
        q_exact = """
            SELECT stop_id, name, ST_X(geom_4326), ST_Y(geom_4326)
            FROM stop 
            WHERE name ILIKE %s
            ORDER BY length(name) ASC 
            LIMIT %s;
        """
        cur.execute(q_exact, (f"%{name}%", limit))
        rows = cur.fetchall()
        
        if rows:
            return [{"stop_id": r[0], "name": r[1], "lon": float(r[2]), "lat": float(r[3])} for r in rows]

        # 2. البحث المرن (Fuzzy) باستخدام similarity لتفادي مشاكل عامل %
        cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
        q_fuzzy = """
            SELECT stop_id, name, ST_X(geom_4326), ST_Y(geom_4326), similarity(name, %s) as score
            FROM stop 
            WHERE similarity(name, %s) > 0.2
            ORDER BY score DESC 
            LIMIT %s;
        """
        cur.execute(q_fuzzy, (name, name, limit))
        rows = cur.fetchall()
        return [{"stop_id": r[0], "name": r[1], "lon": float(r[2]), "lat": float(r[3])} for r in rows]

    except Exception as e:
        print(f"[DB Error] search_stop: {e}")
        return []
    finally:
        if conn: conn.close()

def get_nearest_stop_db(lat: float, lon: float):
    conn = get_db_connection()
    if not conn: return None
    try:
        cur = conn.cursor()
        # يبحث عن أقرب محطة في دائرة 1 كم
        q = """
            SELECT stop_id, name, 
                   ST_Distance(geom_4326::geography, ST_SetSRID(ST_Point(%s, %s), 4326)::geography) as dist_m
            FROM stop 
            WHERE ST_DWithin(geom_4326::geography, ST_SetSRID(ST_Point(%s, %s), 4326)::geography, 1000)
            ORDER BY geom_4326 <-> ST_SetSRID(ST_Point(%s, %s), 4326) LIMIT 1
        """
        cur.execute(q, (lon, lat, lon, lat, lon, lat))
        r = cur.fetchone()
        if r: return {"stop_id": r[0], "name": r[1], "distance_m": float(r[2])}
    except: return None
    finally: conn.close()

def predict_trip_price(route_id, distance_km=5.0):
    if PRICE_MODEL is None: return 5.0
    try:
        features = pd.DataFrame([{
            'route_id': route_id,
            'distance_km': distance_km,
            'stops_count': 10
        }])
        return float(PRICE_MODEL.predict(features)[0])
    except:
        return 5.0

def decode_route_from_db(route_id):
    if str(route_id) == 'WALK': return "مشي"
    conn = get_db_connection()
    name = str(route_id)
    if conn:
        try:
            cur = conn.cursor()
            cur.execute("SELECT name FROM route WHERE route_id = %s", (int(route_id),))
            res = cur.fetchone()
            if res: name = res[0]
        except: pass
        finally: conn.close()
    return name

def find_journeys_db(origin_stop_id: int, dest_stop_id: int, max_results: int = 5) -> List[Journey]:
    """
    يبحث عن مسارات (مباشر، 1 تبديل، 2 تبديل، 3 تبديلات) + مشي.
    ويقوم بالترتيب النهائي لضمان عدم ضياع المسارات الجيدة.
    """
    conn = get_db_connection()
    if not conn: return []
    results = []
    try:
        cur = conn.cursor()
        
        cur.execute("SELECT to_regclass('public.ways');")
        has_routing = cur.fetchone()[0] is not None
        
        walk_query = ""
        walk_params = []
        if has_routing:
            # حساب المشي الحقيقي بـ pgRouting
            walk_query = """
            UNION ALL
            SELECT 'WALK'::text, 0::numeric,
                COALESCE((SELECT SUM(cost) FROM pgr_dijkstra(
                    'SELECT gid as id, source, target, cost, reverse_cost FROM ways',
                    (SELECT id FROM ways_vertices_pgr ORDER BY the_geom <-> (SELECT geom_4326 FROM stop WHERE stop_id = %s) LIMIT 1),
                    (SELECT id FROM ways_vertices_pgr ORDER BY the_geom <-> (SELECT geom_4326 FROM stop WHERE stop_id = %s) LIMIT 1),
                    false
                )), 0)
            """
            walk_params = [origin_stop_id, dest_stop_id]
        
        # استعلام SQL ضخم يجمع كل الاحتمالات (حتى 3 تبديلات)
        q = f"""
            SELECT * FROM (
                -- 0 Transfers (Direct)
                  SELECT r.route_id::text as path, 0::numeric as money, 0::numeric as walk
                FROM route r
                JOIN trip t ON t.route_id = r.route_id
                JOIN route_stop a ON a.trip_id = t.trip_id
                JOIN route_stop b ON b.trip_id = t.trip_id
                WHERE a.stop_id = %s AND b.stop_id = %s AND a.stop_sequence < b.stop_sequence
                
                UNION ALL
                
                -- 1 Transfer
                  SELECT (r1.route_id::text || ',' || r2.route_id::text),
                      0::numeric, 0::numeric
                FROM route_stop a1
                JOIN trip t1 ON a1.trip_id = t1.trip_id
                JOIN route r1 ON t1.route_id = r1.route_id
                JOIN route_stop a2 ON t1.trip_id = a2.trip_id AND a1.stop_sequence < a2.stop_sequence
                JOIN route_stop b1 ON a2.stop_id = b1.stop_id 
                JOIN trip t2 ON b1.trip_id = t2.trip_id
                JOIN route r2 ON t2.route_id = r2.route_id
                JOIN route_stop b2 ON t2.trip_id = b2.trip_id AND b1.stop_sequence < b2.stop_sequence
                WHERE a1.stop_id = %s AND b2.stop_id = %s AND r1.route_id <> r2.route_id
                
                UNION ALL
                
                -- 2 Transfers
                  SELECT (r1.route_id::text || ',' || r2.route_id::text || ',' || r3.route_id::text),
                      0::numeric, 0::numeric
                FROM route_stop a1
                JOIN trip t1 ON a1.trip_id = t1.trip_id
                JOIN route r1 ON t1.route_id = r1.route_id
                JOIN route_stop a2 ON t1.trip_id = a2.trip_id AND a1.stop_sequence < a2.stop_sequence
                JOIN route_stop b1 ON a2.stop_id = b1.stop_id 
                JOIN trip t2 ON b1.trip_id = t2.trip_id
                JOIN route r2 ON t2.route_id = r2.route_id
                JOIN route_stop b2 ON t2.trip_id = b2.trip_id AND b1.stop_sequence < b2.stop_sequence
                JOIN route_stop c1 ON b2.stop_id = c1.stop_id
                JOIN trip t3 ON c1.trip_id = t3.trip_id
                JOIN route r3 ON t3.route_id = r3.route_id
                JOIN route_stop c2 ON t3.trip_id = c2.trip_id AND c1.stop_sequence < c2.stop_sequence
                WHERE a1.stop_id = %s AND c2.stop_id = %s 
                  AND r1.route_id <> r2.route_id AND r2.route_id <> r3.route_id
                
                UNION ALL
                
                -- 3 Transfers
                  SELECT (r1.route_id::text || ',' || r2.route_id::text || ',' || r3.route_id::text || ',' || r4.route_id::text),
                      0::numeric, 0::numeric
                FROM route_stop a1
                JOIN trip t1 ON a1.trip_id = t1.trip_id
                JOIN route r1 ON t1.route_id = r1.route_id
                JOIN route_stop a2 ON t1.trip_id = a2.trip_id AND a1.stop_sequence < a2.stop_sequence
                JOIN route_stop b1 ON a2.stop_id = b1.stop_id 
                JOIN trip t2 ON b1.trip_id = t2.trip_id
                JOIN route r2 ON t2.route_id = r2.route_id
                JOIN route_stop b2 ON t2.trip_id = b2.trip_id AND b1.stop_sequence < b2.stop_sequence
                JOIN route_stop c1 ON b2.stop_id = c1.stop_id
                JOIN trip t3 ON c1.trip_id = t3.trip_id
                JOIN route r3 ON t3.route_id = r3.route_id
                JOIN route_stop c2 ON t3.trip_id = c2.trip_id AND c1.stop_sequence < c2.stop_sequence
                JOIN route_stop d1 ON c2.stop_id = d1.stop_id
                JOIN trip t4 ON d1.trip_id = t4.trip_id
                JOIN route r4 ON t4.route_id = r4.route_id
                JOIN route_stop d2 ON t4.trip_id = d2.trip_id AND d1.stop_sequence < d2.stop_sequence
                WHERE a1.stop_id = %s AND d2.stop_id = %s 
                  AND r1.route_id <> r2.route_id AND r3.route_id <> r4.route_id

                {walk_query}
            ) AS all_routes
            ORDER BY money ASC, walk ASC
            LIMIT %s;
        """
        
        # Params: (Zero*2) + (One*2) + (Two*2) + (Three*2) + [Walk*2] + Limit
        params = [origin_stop_id, dest_stop_id] * 4 
        params += walk_params
        params += [max_results]

        cur.execute(q, tuple(params))
        rows = cur.fetchall()

        for path_str, _, walk_dist in rows:
            if path_str == 'WALK_NO_DATA': continue
            paths = path_str.split(',')
            total_price = 0.0
            
            for p in paths:
                if p == 'WALK': continue
                try:
                    price = predict_trip_price(int(p))
                    total_price += price
                except:
                    total_price += 2.0 
            
            results.append({
                "path": paths,
                "costs": {
                    "money": float(total_price),
                    "walk": float(walk_dist or 0)
                }
            })

    except Exception as e:
        print(f"DB Error: {e}")
    finally:
        if conn: conn.close()
    
    return results

def compute_walk_meters_point_to_stop(lat: float, lon: float, stop_id: int) -> float:
    """يحسب مسافة المشي من نقطة (lat, lon) لأقرب نقطة وصول لمحطة stop_id باستخدام pgRouting إن توفرت.
    fallback: ST_Distance على الجيوديسك."""
    conn = get_db_connection()
    if not conn:
        return 0.0
    try:
        cur = conn.cursor()
        cur.execute("SELECT to_regclass('public.ways');")
        has_routing = cur.fetchone()[0] is not None

        if has_routing:
            q = """
                SELECT COALESCE((SELECT SUM(cost) FROM pgr_dijkstra(
                    'SELECT gid as id, source, target, cost, reverse_cost FROM ways',
                    (SELECT id FROM ways_vertices_pgr ORDER BY the_geom <-> ST_SetSRID(ST_Point(%s, %s), 4326) LIMIT 1),
                    (SELECT id FROM ways_vertices_pgr ORDER BY the_geom <-> (SELECT geom_4326 FROM stop WHERE stop_id = %s) LIMIT 1),
                    false
                )), 0)::float8 AS dist_m;
            """
            cur.execute(q, (lon, lat, stop_id))
            r = cur.fetchone()
            d = float(r[0] or 0.0)
            if d > 1.0:
                return d
            # fallback لو المسار غير متصل
            q2 = """
                SELECT ST_Distance(
                    ST_SetSRID(ST_Point(%s, %s), 4326)::geography,
                    (SELECT geom_4326 FROM stop WHERE stop_id = %s)::geography
                )::float8;
            """
            cur.execute(q2, (lon, lat, stop_id))
            r2 = cur.fetchone()
            return float(r2[0] or 0.0)
        else:
            q = """
                SELECT ST_Distance(
                    ST_SetSRID(ST_Point(%s, %s), 4326)::geography,
                    (SELECT geom_4326 FROM stop WHERE stop_id = %s)::geography
                )::float8;
            """
            cur.execute(q, (lon, lat, stop_id))
            r = cur.fetchone()
            return float(r[0] or 0.0)
    except Exception:
        return 0.0
    finally:
        try:
            conn.close()
        except Exception:
            pass

def filter_best_journeys(journeys: List[Journey], max_results=5) -> List[Journey]:
    return sorted(journeys, key=lambda x: (x["costs"]["money"], x["costs"]["walk"]))[:max_results]