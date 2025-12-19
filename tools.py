from langchain.tools import tool
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import os
import psycopg2
import time
from models.trip_price_class import TripPricePredictor, load_model
from typing import TypedDict, List
import numpy as np
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

# -------- Known Prices (DirectKnownPrices.csv) --------
_KNOWN_PRICE_BY_GTFS: dict = {}
_KNOWN_PRICE_BY_NAME: dict = {}

def _load_known_prices():
    global _KNOWN_PRICE_BY_GTFS, _KNOWN_PRICE_BY_NAME
    try:
        csv_path = os.path.join(os.getcwd(), 'DirectKnownPrices.csv')
        if not os.path.exists(csv_path):
            # try relative to project root if running from subfolder
            alt = os.path.abspath(os.path.join(os.path.dirname(__file__), 'DirectKnownPrices.csv'))
            csv_path = alt if os.path.exists(alt) else csv_path
        df = pd.read_csv(csv_path)
        # normalize
        if 'Price' not in df.columns:
            return
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        df = df.dropna(subset=['Price'])
        # group by GTFS route_id
        if 'route_id' in df.columns:
            g = df.groupby('route_id')['Price'].median()
            _KNOWN_PRICE_BY_GTFS = g.to_dict()
        # group by long name
        name_col = 'route_long_name_x' if 'route_long_name_x' in df.columns else 'route_long_name'
        if name_col in df.columns:
            g2 = df.groupby(name_col)['Price'].median()
            _KNOWN_PRICE_BY_NAME = g2.to_dict()
        print(f"✅ Loaded known prices: {len(_KNOWN_PRICE_BY_GTFS)} by GTFS id, {len(_KNOWN_PRICE_BY_NAME)} by name")
    except Exception as e:
        print(f"⚠️ Warning: Could not load DirectKnownPrices.csv: {e}")

_load_known_prices()

def get_route_identity(route_id: int) -> dict:
    """Fetch GTFS route id and name for a DB route_id."""
    conn = get_db_connection()
    if not conn:
        return {'gtfs_route_id': None, 'name': None}
    try:
        cur = conn.cursor()
        cur.execute("SELECT gtfs_route_id, name FROM route WHERE route_id = %s", (route_id,))
        r = cur.fetchone()
        if not r:
            return {'gtfs_route_id': None, 'name': None}
        return {'gtfs_route_id': r[0], 'name': r[1]}
    except Exception:
        return {'gtfs_route_id': None, 'name': None}
    finally:
        try:
            conn.close()
        except Exception:
            pass

def get_known_price_for_route(route_id: int) -> float | None:
    """Return known price for this route if present in DirectKnownPrices.csv."""
    ident = get_route_identity(route_id)
    gtfs_id = ident.get('gtfs_route_id')
    name = ident.get('name')
    if gtfs_id and gtfs_id in _KNOWN_PRICE_BY_GTFS:
        return float(_KNOWN_PRICE_BY_GTFS[gtfs_id])
    if name and name in _KNOWN_PRICE_BY_NAME:
        return float(_KNOWN_PRICE_BY_NAME[name])
    return None

def search_stop_by_name_db(name: str, limit: int = 5):
    
    conn = get_db_connection()
    if not conn: return []
    try:
        cur = conn.cursor()
        
        # Exact match first
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

        # Fuzzy search fallback
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
        
        # Find nearest stop within 1km
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

def predict_trip_price(distance_km: float) -> float:
#predict price using only distance (log-transformed) with the loaded model
    if PRICE_MODEL is None:
        raise RuntimeError("Price model not loaded")
    return float(PRICE_MODEL.predict([distance_km])[0])

def get_route_cost_db(route_id: int) -> float:
    #read known cost for route from route table as fallback if model not available
    conn = get_db_connection()
    if not conn:
        return 0.0
    try:
        cur = conn.cursor()
        cur.execute("SELECT cost FROM route WHERE route_id = %s", (route_id,))
        r = cur.fetchone()
        if not r:
            return 0.0
        return float(r[0] or 0.0)
    except Exception:
        return 0.0
    finally:
        try:
            conn.close()
        except Exception:
            pass

def get_route_details_db(route_id: int) -> dict:
    """Fetch route details: distance_km, stops_count, route_type (mode)"""
    conn = get_db_connection()
    if not conn:
        return {"distance_km": 0.0, "stops_count": 0, "route_type": "unknown"}
    try:
        cur = conn.cursor()

        # route_type (mode)
        cur.execute("SELECT COALESCE(mode, 'unknown') FROM route WHERE route_id = %s", (route_id,))
        r_mode = cur.fetchone()
        route_type = (r_mode[0] if r_mode else 'unknown') or 'unknown'

        # Prefer geometry length if available
        cur.execute(
            """
            SELECT COALESCE(ST_Length(geom_22992), ST_Length(ST_Transform(geom_4326,22992)))::float8
            FROM route_geometry WHERE route_id = %s
            ORDER BY route_geom_id ASC LIMIT 1
            """,
            (route_id,)
        )
        r_len = cur.fetchone()
        length_m = float(r_len[0]) if r_len and r_len[0] is not None else None

        # Find a sample trip on this route
        cur.execute("SELECT trip_id FROM trip WHERE route_id = %s ORDER BY trip_id ASC LIMIT 1", (route_id,))
        r_trip = cur.fetchone()
        sample_trip_id = int(r_trip[0]) if r_trip else None

        stops_count = 0
        if sample_trip_id:
            cur.execute("SELECT COUNT(*) FROM route_stop WHERE trip_id = %s", (sample_trip_id,))
            stops_count = int(cur.fetchone()[0])

        # If no geometry length, approximate by summing adjacent stop distances for the sample trip
        if length_m is None and sample_trip_id:
            cur.execute(
                """
                SELECT SUM(ST_Distance(s1.geom_22992, s2.geom_22992))::float8
                FROM route_stop rs1
                JOIN route_stop rs2 ON rs2.trip_id = rs1.trip_id AND rs2.stop_sequence = rs1.stop_sequence + 1
                JOIN stop s1 ON s1.stop_id = rs1.stop_id
                JOIN stop s2 ON s2.stop_id = rs2.stop_id
                WHERE rs1.trip_id = %s
                """,
                (sample_trip_id,)
            )
            r_sum = cur.fetchone()
            length_m = float(r_sum[0]) if r_sum and r_sum[0] is not None else 0.0

        distance_km = float(length_m or 0.0) / 1000.0
        return {"distance_km": distance_km, "stops_count": stops_count, "route_type": route_type}
    except Exception:
        return {"distance_km": 0.0, "stops_count": 0, "route_type": "unknown"}
    finally:
        try:
            conn.close()
        except Exception:
            pass

def predict_trip_price_from_features(distance_km: float, stops_count: int, route_type: str) -> float:
    #predict price using actual route features with log-transformed distance
    if PRICE_MODEL is None:
        raise RuntimeError("Price model not loaded")
    # Prepare features; include both raw and log distance to be robust to training config
    distance_km = max(float(distance_km), 0.0)
    features = pd.DataFrame([
        {
            'distance_km': distance_km,
            'distance_km_log': float(np.log1p(distance_km)),
            'stops_count': int(stops_count or 0),
            'route_type': (route_type or 'unknown')
        }
    ])
    try:
        # Use underlying sklearn model directly, then apply bus rounding style
        raw = PRICE_MODEL.model.predict(features)
        return float(PRICE_MODEL._round_bus_style(raw)[0])
    except Exception:
        # Fallback to wrapper single-distance predict if pipeline expects only distance
        return float(PRICE_MODEL.predict([distance_km])[0])

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
    return arabize_text(name)

def find_journeys_db(origin_stop_id: int, dest_stop_id: int, max_results: int = 5) -> List[Journey]:
    # Find journey options between two stop IDs from the DB
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
            # Calculate real walking distance using pgRouting
            walk_query = """
            UNION ALL
            SELECT 'WALK'::text, 0::numeric,
                COALESCE((SELECT SUM(cost) FROM pgr_dijkstra(
                    'SELECT gid as id, source, target, cost, reverse_cost FROM ways',
                    (SELECT id FROM ways_vertices_pgr ORDER BY the_geom <-> (SELECT geom_4326 FROM stop WHERE stop_id = %s) LIMIT 1),
                    (SELECT id FROM ways_vertices_pgr ORDER BY the_geom <-> (SELECT geom_4326 FROM stop WHERE stop_id = %s) LIMIT 1),
                    false
                )), 0),
                (%s::text || ',' || %s::text) AS stops_path
            """
            walk_params = [origin_stop_id, dest_stop_id, origin_stop_id, dest_stop_id]
        
        # Large SQL query combining all possibilities (up to 3 transfers)
        internal_limit = max(max_results * 10, 50)
        q = f"""
            SELECT * FROM (
                -- 0 Transfers (Direct)
            SELECT r.route_id::text as path, r.cost::numeric as money, 0::numeric as walk,
                   (a.stop_id::text || ',' || b.stop_id::text) AS stops_path
                FROM route r
                JOIN trip t ON t.route_id = r.route_id
                JOIN route_stop a ON a.trip_id = t.trip_id
                JOIN route_stop b ON b.trip_id = t.trip_id
                WHERE a.stop_id = %s AND b.stop_id = %s AND a.stop_sequence < b.stop_sequence
                
                UNION ALL
                
                -- 1 Transfer
            SELECT (r1.route_id::text || ',' || r2.route_id::text),
                   (r1.cost + r2.cost)::numeric, 0::numeric,
                   (a1.stop_id::text || ',' || a2.stop_id::text || ',' || b2.stop_id::text) AS stops_path
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
                                             (r1.cost + r2.cost + r3.cost)::numeric, 0::numeric,
                                             (a1.stop_id::text || ',' || a2.stop_id::text || ',' || c1.stop_id::text || ',' || c2.stop_id::text) AS stops_path
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
                                             (r1.cost + r2.cost + r3.cost + r4.cost)::numeric, 0::numeric,
                                             (a1.stop_id::text || ',' || a2.stop_id::text || ',' || c1.stop_id::text || ',' || d1.stop_id::text || ',' || d2.stop_id::text) AS stops_path
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
            LIMIT {internal_limit};
        """
        
        # Params: (Zero*2) + (One*2) + (Two*2) + (Three*2) + [Walk*2] + Limit
        params = [origin_stop_id, dest_stop_id] * 4 
        params += walk_params
        # no final limit param used; we set higher internal limit above

        cur.execute(q, tuple(params))
        rows = cur.fetchall()

        seen = set()
        for path_str, money_db, walk_dist, stops_path in rows:
            if path_str == 'WALK_NO_DATA': continue
            paths = path_str.split(',')
            # Deduplicate identical path combos
            key = tuple(paths)
            if key in seen:
                continue
            seen.add(key)
            total_price = 0.0
            stop_ids = [int(s) for s in stops_path.split(',') if s]
            
            for p in paths:
                if p == 'WALK':
                    continue
                rid = int(p)
                # 1) prefer known price from CSV
                known = get_known_price_for_route(rid)
                if known is not None:
                    total_price += float(known)
                    continue
                # 2) else use model on real features
                try:
                    details = get_route_details_db(rid)
                    price = predict_trip_price_from_features(
                        details['distance_km'], details['stops_count'], details['route_type']
                    )
                    total_price += float(price or 0.0)
                except Exception:
                    # 3) fallback to DB route cost
                    total_price += float(get_route_cost_db(rid) or 0.0)
            
            results.append({
                "path": paths,
                "stops_path": stop_ids,
                "costs": {
                    "money": float(total_price),
                    "walk": float(walk_dist or 0)
                }
            })

    except Exception as e:
        print(f"DB Error: {e}")
    finally:
        if conn: conn.close()
    
    # Sort by price then walk, and cap to requested max_results
    results = sorted(results, key=lambda x: (x["costs"]["money"], x["costs"]["walk"]))[:max_results]
    return results

def arabize_text(text: str) -> str:
    #arabize common route names for better final response display
    if not text:
        return text
    mapping = {
        'Asafra': 'العصافرة',
        'El-Mandara': 'المندرة',
        'El-Mawqaf El-Geded': 'الموقف الجديد',
        'El-Mawqaf Geded': 'الموقف الجديد',
        'El-Mansheya': 'المنشية',
        'Raml Station': 'محطة الرمل',
        'San Stefano': 'سان ستيفانو',
        'Victoria': 'فيكتوريا',
        'Gleem': 'جليم',
        'Stanley': 'ستانلي',
        'Miami': 'ميامي',
        'Sidi Bishr': 'سيدي بشر',
        'Sidi Gabir': 'سيدي جابر',
        'Abu Qir': 'ابو قير',
        'Kilo 21': 'الكيلو 21',
        'Agamy': 'العجمي'
    }
    parts = [p.strip() for p in text.split(' - ')]
    arabized = []
    for p in parts:
        arabized.append(mapping.get(p, p))
    return ' - '.join(arabized)

def compute_walk_meters_point_to_stop(lat: float, lon: float, stop_id: int) -> float:
    #
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

def compute_path_meters_between_stops(stop_a_id: int, stop_b_id: int) -> float:
    
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
                    (SELECT id FROM ways_vertices_pgr ORDER BY the_geom <-> (SELECT geom_4326 FROM stop WHERE stop_id = %s) LIMIT 1),
                    (SELECT id FROM ways_vertices_pgr ORDER BY the_geom <-> (SELECT geom_4326 FROM stop WHERE stop_id = %s) LIMIT 1),
                    false
                )), 0)::float8 AS dist_m;
            """
            cur.execute(q, (stop_a_id, stop_b_id))
            r = cur.fetchone()
            d = float(r[0] or 0.0)
            if d > 1.0:
                return d
            # Fallback to geodesic distance
            q2 = """
                SELECT ST_Distance(
                    (SELECT geom_4326 FROM stop WHERE stop_id = %s)::geography,
                    (SELECT geom_4326 FROM stop WHERE stop_id = %s)::geography
                )::float8;
            """
            cur.execute(q2, (stop_a_id, stop_b_id))
            r2 = cur.fetchone()
            return float(r2[0] or 0.0)
        else:
            q = """
                SELECT ST_Distance(
                    (SELECT geom_4326 FROM stop WHERE stop_id = %s)::geography,
                    (SELECT geom_4326 FROM stop WHERE stop_id = %s)::geography
                )::float8;
            """
            cur.execute(q, (stop_a_id, stop_b_id))
            r = cur.fetchone()
            return float(r[0] or 0.0)
    except Exception:
        return 0.0
    finally:
        try:
            conn.close()
        except Exception:
            pass

def compute_walk_meters_point_to_point(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    
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
                    (SELECT id FROM ways_vertices_pgr ORDER BY the_geom <-> ST_SetSRID(ST_Point(%s, %s), 4326) LIMIT 1),
                    false
                )), 0)::float8 AS dist_m;
            """
            cur.execute(q, (lon1, lat1, lon2, lat2))
            r = cur.fetchone()
            d = float(r[0] or 0.0)
            if d > 1.0:
                return d
        # Fallback geodesic
        q2 = """
            SELECT ST_Distance(
                ST_SetSRID(ST_Point(%s, %s), 4326)::geography,
                ST_SetSRID(ST_Point(%s, %s), 4326)::geography
            )::float8;
        """
        cur.execute(q2, (lon1, lat1, lon2, lat2))
        r2 = cur.fetchone()
        return float(r2[0] or 0.0)
    except Exception:
        return 0.0
    finally:
        try:
            conn.close()
        except Exception:
            pass

def filter_best_journeys(journeys: List[Journey], max_results=5) -> List[Journey]:
    return sorted(journeys, key=lambda x: (x["costs"]["money"], x["costs"]["walk"]))[:max_results]