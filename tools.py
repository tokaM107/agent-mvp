from langchain.tools import tool
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import os
import psycopg2
import time
from models.trip_price_class import TripPricePredictor, load_model, TripPricePreedictor
from typing import TypedDict, List
import pandas as pd

# Load Pricing Model Once
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
    conn = get_db_connection()
    if not conn: return []
    try:
        cur = conn.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
        q = """
            SELECT stop_id, name, ST_X(geom_4326), ST_Y(geom_4326), similarity(name, %s) as score
            FROM stop WHERE name %% %s ORDER BY score DESC LIMIT %s
        """
        cur.execute(q, (name, name, limit))
        return [{"stop_id": r[0], "name": r[1], "lon": float(r[2]), "lat": float(r[3])} for r in cur.fetchall()]
    except: return []
    finally: conn.close()

def get_nearest_stop_db(lat: float, lon: float):
    conn = get_db_connection()
    if not conn: return None
    try:
        cur = conn.cursor()
        # Get nearest stop AND its distance in meters
        q = """
            SELECT stop_id, name, 
                   ST_Distance(geom_4326::geography, ST_SetSRID(ST_Point(%s, %s), 4326)::geography) as dist_m
            FROM stop 
            ORDER BY geom_4326 <-> ST_SetSRID(ST_Point(%s, %s), 4326) LIMIT 1
        """
        cur.execute(q, (lon, lat, lon, lat))
        r = cur.fetchone()
        if r: return {"stop_id": r[0], "name": r[1], "distance_m": float(r[2])}
    except: return None
    finally: conn.close()

def predict_trip_price(route_id, distance_km=5.0):
    """Calculates price using the Python ML Model"""
    if PRICE_MODEL is None: return 5.0 # Fallback
    try:
        # Create a dummy DataFrame with features expected by the model
        # You might need to adjust features based on your specific model training
        features = pd.DataFrame([{
            'route_id': route_id,
            'distance_km': distance_km,
            'stops_count': 10 # approximate
        }])
        # Ensure columns match what the model expects
        return float(PRICE_MODEL.predict(features)[0])
    except:
        return 5.0 # Fallback default

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
    conn = get_db_connection()
    if not conn: return []
    results = []
    try:
        cur = conn.cursor()
        
        # 1. Check for pgRouting functionality
        cur.execute("SELECT to_regclass('public.ways');")
        has_routing = cur.fetchone()[0] is not None
        
        walk_query = ""
        if has_routing:
            # Calculate REAL walking distance between stops using pgRouting
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
        else:
            walk_query = "UNION ALL SELECT 'WALK_NO_DATA', 0, 0 WHERE false"
            walk_params = []

        # 2. SQL Query (Fixed Joins & Logic)
        q = f"""
            WITH zero AS (
                SELECT r.route_id::text as path, 0::numeric as money, 0::numeric as walk
                FROM route r
                JOIN trip t ON t.route_id = r.route_id
                JOIN route_stop a ON a.trip_id = t.trip_id
                JOIN route_stop b ON b.trip_id = t.trip_id
                WHERE a.stop_id = %s AND b.stop_id = %s AND a.stop_sequence < b.stop_sequence
                LIMIT 3
            ),
            one AS (
                SELECT (r1.route_id::text || ',' || r2.route_id::text) as path,
                       0::numeric as money, 100::numeric as walk -- 100m approx transfer
                FROM route_stop a1
                JOIN trip t1 ON a1.trip_id = t1.trip_id
                JOIN route r1 ON t1.route_id = r1.route_id
                JOIN route_stop a2 ON t1.trip_id = a2.trip_id AND a1.stop_sequence < a2.stop_sequence
                JOIN route_stop b1 ON a2.stop_id = b1.stop_id -- Transfer
                JOIN trip t2 ON b1.trip_id = t2.trip_id
                JOIN route r2 ON t2.route_id = r2.route_id
                JOIN route_stop b2 ON t2.trip_id = b2.trip_id AND b1.stop_sequence < b2.stop_sequence
                WHERE a1.stop_id = %s AND b2.stop_id = %s AND r1.route_id <> r2.route_id
                LIMIT 3
            )
            SELECT path, money, walk FROM zero
            UNION ALL
            SELECT path, money, walk FROM one
            {walk_query}
            LIMIT %s;
        """
        
        params = [origin_stop_id, dest_stop_id] + [origin_stop_id, dest_stop_id] + walk_params + [max_results]
        cur.execute(q, tuple(params))
        rows = cur.fetchall()

        for path_str, _, walk_dist in rows:
            if path_str == 'WALK_NO_DATA': continue
            
            paths = path_str.split(',')
            total_price = 0.0
            
            # --- CALCULATE PRICE USING PYTHON MODEL ---
            for p in paths:
                if p == 'WALK': continue
                try:
                    # Predict price for this route (assume avg 5km trip if unknown)
                    price = predict_trip_price(int(p), distance_km=5.0)
                    total_price += price
                except:
                    total_price += 2.0 # Default fallback
            
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

def filter_best_journeys(journeys: List[Journey], max_results=5) -> List[Journey]:
    # Sort by Money then Walk
    return sorted(journeys, key=lambda x: (x["costs"]["money"], x["costs"]["walk"]))[:max_results]

# Note: format_journeys_for_user removed from here to let Gemini do it