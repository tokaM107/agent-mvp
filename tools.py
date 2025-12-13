from langchain.tools import tool
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import os
import psycopg2
import time
import osmnx as ox
from models.trip_price_class import TripPricePredictor
from models.trip_price_class import load_model
from graph_builder import attach_trips_to_graph
import heapq
from collections import deque, defaultdict
from services.pricing import get_cost
from trip_decoder import decode_trip
from typing import TypedDict, List

class JourneyCosts(TypedDict):
    money: float
    walk: float

class Journey(TypedDict):
    path: List[str]
    costs: JourneyCosts


GLOBAL_G = None

def set_graph(g):
    global GLOBAL_G
    GLOBAL_G = g


# Removed GTFS-derived aliasing: geocoding uses Nominatim only


def geocode_address(address: str) -> dict:
    """Geocode an address to latitude/longitude using Nominatim only,
    with Alexandria, Egypt locality bias for improved matching."""
    geolocator = Nominatim(user_agent="alex-transport-agent")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=0.8)
    # Ensure locality context for better accuracy
    query = address.strip()
    if ("Alexandria" not in query) and ("الإسكندرية" not in query) and ("Alexandria, Egypt" not in query):
        query = f"{query}, Alexandria, Egypt"
    location = geocode(query)
    if location:
        return {"lat": float(location.latitude), "lon": float(location.longitude)}
    else:
        return {"error": "Location not found"}


# --- Database helpers (optional PostGIS integration) ---
def get_db_connection():
    """Create a connection to the local transport_db.
    Controlled by environment variables:
    DB_HOST, DB_NAME, DB_USER, DB_PASSWORD
    Defaults: localhost, transport_db, postgres, postgres
    """
    host = os.environ.get("DB_HOST", "localhost")
    db = os.environ.get("DB_NAME", "transport_db")
    user = os.environ.get("DB_USER", "postgres")
    pwd = os.environ.get("DB_PASSWORD", "postgres")
    try:
        return psycopg2.connect(host=host, database=db, user=user, password=pwd)
    except Exception:
        return None


def snap_stop_to_vertex_db(stop_id: int):
    """Find nearest pgRouting vertex (ways_vertices_pgr) to a stop.
    Returns vertex_id (ways_vertices_pgr.id) or None.
    """
    conn = get_db_connection()
    if not conn:
        return None
    try:
        cur = conn.cursor()
        q = (
            "SELECT v.id FROM ways_vertices_pgr v, stop s "
            "WHERE s.stop_id = %s "
            "ORDER BY v.the_geom <-> s.geom_4326 LIMIT 1;"
        )
        cur.execute(q, (stop_id,))
        row = cur.fetchone()
        return row[0] if row else None
    except Exception:
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass


def compute_walk_distance_db(stop_id_a: int, stop_id_b: int):
    """Compute walking distance between two stops using pgr_dijkstra on ways table.
    Returns distance in meters or None if no path found.
    """
    conn = get_db_connection()
    if not conn:
        return None
    try:
        vertex_a = snap_stop_to_vertex_db(stop_id_a)
        vertex_b = snap_stop_to_vertex_db(stop_id_b)
        if not vertex_a or not vertex_b:
            return None
        
        cur = conn.cursor()
        # pgr_dijkstra(edges_sql, start_vid, end_vid, directed)
        # ways table has: id, source, target, cost (length in meters), reverse_cost
        q = (
            "SELECT SUM(edge.cost) AS total_distance "
            "FROM pgr_dijkstra("
            "  'SELECT gid AS id, source, target, cost, reverse_cost FROM ways', "
            "  %s, %s, directed:=false"
            ") AS route "
            "JOIN ways AS edge ON route.edge = edge.gid;"
        )
        cur.execute(q, (vertex_a, vertex_b))
        row = cur.fetchone()
        return float(row[0]) if row and row[0] else None
    except Exception as e:
        print(f"[ERROR] compute_walk_distance_db: {e}")
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass


def search_stop_by_name_db(name: str, limit: int = 5):
    """Search stops by name using ILIKE with trigram index (if available).
    Returns list of {stop_id, name, lon, lat} from operational `stop` table.
    """
    conn = get_db_connection()
    if not conn:
        return []
    try:
        cur = conn.cursor()
        q = (
            "SELECT stop_id, name, ST_X(geom_4326), ST_Y(geom_4326) "
            "FROM stop WHERE name ILIKE %s LIMIT %s;"
        )
        cur.execute(q, (f"%{name}%", limit))
        rows = cur.fetchall()
        return [{"stop_id": r[0], "name": r[1], "lon": float(r[2]), "lat": float(r[3])} for r in rows]
    except Exception:
        return []
    finally:
        try:
            conn.close()
        except Exception:
            pass


def get_nearest_stop_db(lat: float, lon: float):
    """Use PostGIS to find nearest stop to a coordinate in meters.
    Requires operational `stop` with `geom_4326` geometry (SRID 4326) and projected `geom_22992` if available.
    """
    conn = get_db_connection()
    if not conn:
        return None
    try:
        cur = conn.cursor()
        # Prefer KNN operator <-> on WGS84; or use projected distance if geom_22992 exists
        q = (
            "SELECT stop_id, name, ST_Distance(geom_4326, ST_SetSRID(ST_Point(%s, %s), 4326)) AS distance "
            "FROM stop "
            "ORDER BY geom_4326 <-> ST_SetSRID(ST_Point(%s, %s), 4326) "
            "LIMIT 1;"
        )
        cur.execute(q, (lon, lat, lon, lat))
        r = cur.fetchone()
        if r:
            return {"stop_id": r[0], "name": r[1], "distance_m": float(r[2])}
        return None
    except Exception:
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _get_routes_passing_stop(conn, stop_id: int, limit: int = 100):
    cur = conn.cursor()
    q = (
        "SELECT r.route_id, r.name, rs.stop_sequence "
        "FROM route_stop rs JOIN route r ON rs.route_id = r.route_id "
        "WHERE rs.stop_id = %s "
        "ORDER BY r.name, rs.stop_sequence LIMIT %s;"
    )
    cur.execute(q, (stop_id, limit))
    rows = cur.fetchall()
    return rows  # (route_id, route_name, stop_sequence)


def _get_common_transfer_stops(conn, route_id_a: int, route_id_b: int, limit: int = 20):
    cur = conn.cursor()
    q = (
        "SELECT s.stop_id, s.name "
        "FROM route_stop rsa "
        "JOIN route_stop rsb ON rsa.stop_id = rsb.stop_id "
        "JOIN stop s ON s.stop_id = rsa.stop_id "
        "WHERE rsa.route_id = %s AND rsb.route_id = %s "
        "GROUP BY s.stop_id, s.name LIMIT %s;"
    )
    cur.execute(q, (route_id_a, route_id_b, limit))
    return cur.fetchall()  # (stop_id, name)


def find_journeys_db(origin_stop_id: int, dest_stop_id: int, max_results: int = 5) -> List[Journey]:
    """Compute simple journeys directly from DB routes:
    - 0-transfer: same route serves both origin and destination in order.
    - 1-transfer: route A from origin to transfer stop + route B from transfer stop to destination.
    Returns minimal set of journeys with rough costs and walking=0 (DB-only stage).
    """
    conn = get_db_connection()
    if not conn:
        return []
    results: List[Journey] = []
    try:
        cur = conn.cursor()
        # 0-transfer: same route contains both stops with correct sequence order
        q0 = (
            "SELECT r.route_id, r.name, a.stop_sequence AS seq_o, b.stop_sequence AS seq_d "
            "FROM route r "
            "JOIN route_stop a ON a.route_id = r.route_id "
            "JOIN route_stop b ON b.route_id = r.route_id "
            "WHERE a.stop_id = %s AND b.stop_id = %s AND a.stop_sequence < b.stop_sequence "
            "ORDER BY r.name LIMIT %s;"
        )
        cur.execute(q0, (origin_stop_id, dest_stop_id, max_results))
        for route_id, route_name, seq_o, seq_d in cur.fetchall():
            # Money estimate: use pricing service if available, else flat cost
            money = 4.0
            path = [str(route_id)]
            results.append({"path": path, "costs": {"money": money, "walk": 0.0}})

        # 1-transfer: origin route to transfer stop, then transfer stop to destination route
        # Get routes serving origin and destination
        routes_o = _get_routes_passing_stop(conn, origin_stop_id)
        routes_d = _get_routes_passing_stop(conn, dest_stop_id)
        routes_o_ids = [(r[0], r[1]) for r in routes_o]
        routes_d_ids = [(r[0], r[1]) for r in routes_d]

        for rid_o, name_o in routes_o_ids[:10]:
            for rid_d, name_d in routes_d_ids[:10]:
                # Skip trivial same-route, already covered
                if rid_o == rid_d:
                    continue
                transfers = _get_common_transfer_stops(conn, rid_o, rid_d, limit=5)
                for t_stop_id, t_name in transfers:
                    # Verify order along origin route: origin -> transfer
                    cur.execute(
                        "SELECT a.stop_sequence AS seq_o, b.stop_sequence AS seq_t FROM route_stop a JOIN route_stop b ON a.route_id=b.route_id WHERE a.route_id=%s AND a.stop_id=%s AND b.stop_id=%s;",
                        (rid_o, origin_stop_id, t_stop_id),
                    )
                    row = cur.fetchone()
                    if not row:
                        continue
                    seq_o, seq_t = row
                    if seq_o is None or seq_t is None or seq_o >= seq_t:
                        continue

                    # Verify order along destination route: transfer -> destination
                    cur.execute(
                        "SELECT a.stop_sequence AS seq_t, b.stop_sequence AS seq_d FROM route_stop a JOIN route_stop b ON a.route_id=b.route_id WHERE a.route_id=%s AND a.stop_id=%s AND b.stop_id=%s;",
                        (rid_d, t_stop_id, dest_stop_id),
                    )
                    row2 = cur.fetchone()
                    if not row2:
                        continue
                    seq_t2, seq_d = row2
                    if seq_t2 is None or seq_d is None or seq_t2 >= seq_d:
                        continue

                    # Compute walking distance at transfer point (if available)
                    walk_dist = 0.0
                    # Optional: calculate walk from origin to first stop or transfer walking
                    # For now, set minimal walk for same-stop transfers
                    # You can enhance with compute_walk_distance_db(origin_stop_id, t_stop_id)
                    
                    # Money estimate: two segments
                    money = 8.0
                    path = [str(rid_o), str(rid_d)]
                    results.append({"path": path, "costs": {"money": money, "walk": walk_dist}})

                    if len(results) >= max_results:
                        break
                if len(results) >= max_results:
                    break
            if len(results) >= max_results:
                break
    except Exception:
        return results
    finally:
        try:
            conn.close()
        except Exception:
            pass
    return results
    
# @tool 
def get_nearest_node(lat: float, lon: float) -> int:
    """Get the nearest OSM node to the given latitude and longitude coordinates."""
    if GLOBAL_G is None:
        raise ValueError("Graph is not initialized. Call set_graph(G) first.")
    node_id = ox.distance.nearest_nodes(GLOBAL_G, lon, lat)
    return int(node_id)

# @tool
def explore_trips(source: int, cutoff=15000) -> dict:
    """
    Returns:
        dict of trip_id -> {
            'stop_id': gtfs_stop_id, # The actual GTFS ID needed for costs
            'osm_node_id': osm_node,       # The physical location
            'walk': distance_m,
            'path': [...]
        }
    """
    if GLOBAL_G is None:
        raise ValueError("Graph is not initialized. Call set_graph(G) first.")
    
    dist = {source: 0.0}
    prev = {}
    pq = [(0.0, source)]
    visited = set()

    trips = {}

    def reconstruct_path(node):
        path = []
        while node in prev:
            path.append(node)
            node = prev[node]
        path.append(source)
        return list(reversed(path))

    while pq:
        d, node = heapq.heappop(pq)

        if d > cutoff:
            break
        if node in visited:
            continue
        visited.add(node)

        # [CHANGE]: Retrieve the boarding_map instead of just a list of trips
        boarding_map = GLOBAL_G.nodes[node].get("boarding_map")
        
        if boarding_map:
            # Iterate over the trips available at this node
            for trip_id, real_stop_id in boarding_map.items():
                
                # Check if this is the best walk to this trip so far
                best = trips.get(trip_id)
                if best is None or d < best["walk"]:
                    trips[trip_id] = {
                        "stop_id": real_stop_id,  # GTFS stop id for pricing
                        "osm_node_id": node,      # OSM node id (kept minimal)
                        "walk": d,                # walking distance in meters
                        # Keep payload lean to reduce LLM tokens
                        "path": []
                    }

        # Relax neighbors (Standard Dijkstra)
        for nbr, edge_data in GLOBAL_G[node].items():
            for _, attr in edge_data.items():
                length = float(attr.get("length", 1.0))
                new_dist = d + length

                if new_dist <= cutoff and new_dist < dist.get(nbr, float("inf")):
                    dist[nbr] = new_dist
                    prev[nbr] = node
                    heapq.heappush(pq, (new_dist, nbr))

    return trips

def set_trip_graph(g, p):
    global TRIP_GRAPH, PATHWAYS_DICT
    TRIP_GRAPH = g
    PATHWAYS_DICT = p



# @tool
def find_journeys(start_trips, goal_trips, max_transfers=2) -> List[Journey]:
    """
    Docstring for find_journeys
    
    :param graph: Description
    :param pathways_dict: Description
    :param start_trips: Description
    :param goal_trips: Description
    :param max_transfers: Description
    """
    graph = TRIP_GRAPH
    pathways_dict = PATHWAYS_DICT
    results = []
    # (current_trip_id, current_board_stop_id, path_list, cumulative_costs)
    queue = deque()
    
    # Pruning dictionary
    best_costs_to_node = defaultdict(lambda: {
        'money': float('inf'),
        # 'transport_time': float('inf'),
        'walk': float('inf')
    })

    # --- 1. Initialize Start Trips ---
    for start_trip_id, data in start_trips.items():
        costs = {
            'money': 0,
            # 'transport_time': 0,
            'walk': data['walk']
        }
        path = [start_trip_id]
        start_stop = data['stop_id']
        
        queue.append((start_trip_id, start_stop, path, costs))
        best_costs_to_node[start_trip_id] = costs.copy()

        # Check 0-transfer goal
        if start_trip_id in goal_trips:
            goal_stop = goal_trips[start_trip_id]['stop_id']
            
            leg_money = get_cost(start_trip_id, start_stop, goal_stop)
            # leg_time = get_transport_time(start_trip_id, start_stop, goal_stop)
            
            final_costs = costs.copy()
            final_costs['money'] += leg_money
            # final_costs['transport_time'] += leg_time
            final_costs['walk'] += goal_trips[start_trip_id]['walk']
            
            results.append({
                "path": path,
                "costs": final_costs
            })

    # --- 2. BFS ---
    while queue:
        (current_trip, current_board_stop, path, current_costs) = queue.popleft()
        
        if len(path) - 1 >= max_transfers:
            continue

        for next_trip, pathway_id in graph.get(current_trip, {}).items():
            pathway = pathways_dict[pathway_id]
            
            if next_trip in path: continue

            # --- Cost Logic ---
            # 1. Transfer Walk
            transfer_walk_cost = pathway['walking_distance_m']
            
            # 2. Cost of the PREVIOUS trip segment (from board_stop to transfer_stop)
            # We are getting off current_trip at pathway['start_stop_id']
            prev_trip_money = get_cost(
                current_trip, 
                current_board_stop, 
                pathway['start_stop_id']
            )
            # prev_trip_time = get_transport_time(
            #     current_trip, 
            #     current_board_stop, 
            #     pathway['start_stop_id']
            # )

            new_costs = {
                'money': current_costs['money'] + prev_trip_money,
                # 'transport_time': current_costs['transport_time'] + prev_trip_time,
                'walk': current_costs['walk'] + transfer_walk_cost
            }

            # --- Pruning---
            best_known = best_costs_to_node[next_trip]
            # if we are better in ANY metric, we explore
            is_potentially_useful = (
                new_costs['money'] < best_known['money'] or
                # new_costs['transport_time'] < best_known['transport_time'] or
                new_costs['walk'] < best_known['walk']
            )

            if is_potentially_useful:
                # Update best knowns
                best_costs_to_node[next_trip]['money'] = min(best_known['money'], new_costs['money'])
                # best_costs_to_node[next_trip]['transport_time'] = min(best_known['transport_time'], new_costs['transport_time'])
                best_costs_to_node[next_trip]['walk'] = min(best_known['walk'], new_costs['walk'])

                new_path = path + [next_trip]
                
                # We board the NEXT trip at pathway['end_stop_id']
                queue.append((next_trip, pathway['end_stop_id'], new_path, new_costs))

                # --- 3. Check Goal ---
                if next_trip in goal_trips:
                    goal_stop = goal_trips[next_trip]['stop_id']
                    
                    # FINAL leg cost (from transfer-in to goal-stop)
                    last_leg_money = get_cost(next_trip, pathway['end_stop_id'], goal_stop)
                    # last_leg_time = get_transport_time(next_trip, pathway['end_stop_id'], goal_stop)
                    
                    final_journey_costs = new_costs.copy()
                    final_journey_costs['money'] += last_leg_money
                    # final_journey_costs['transport_time'] += last_leg_time
                    final_journey_costs['walk'] += goal_trips[next_trip]['walk']
                    
                    results.append({
                        "path": new_path,
                        "costs": final_journey_costs
                    })
    return results

# @tool
def filter_best_journeys(journeys: List[Journey], max_results=5) -> List[Journey]:
    """ Filters and returns the best journeys based on smallest walking distance and monetary cost."""
    sorted_journeys = sorted(
        journeys,
        key=lambda x: (x["costs"]["walk"], x["costs"]["money"])
    )
    return sorted_journeys[:max_results]



# @tool
def format_journeys_for_user(journeys: List[Journey]) -> str:
    """
    Takes raw journeys output and returns user-friendly Arabic description.
    """
    output = ""

    if not journeys:
        return "للأسف، مفيش مسارات مناسبة من مكان الانطلاق للمكان المطلوب حسب البيانات الحالية."

    # Compute metrics for tagging
    monies = [j["costs"]["money"] for j in journeys]
    walks = [j["costs"]["walk"] for j in journeys]
    transfers = [max(0, len(j["path"]) - 1) for j in journeys]

    min_money = min(monies)
    max_money = max(monies)
    min_walk = min(walks)
    min_transfers = min(transfers)

    # Best overall by (money, transfers, walk)
    best_idx = 0
    best_tuple = (monies[0], transfers[0], walks[0])
    for idx in range(1, len(journeys)):
        t = (monies[idx], transfers[idx], walks[idx])
        if t < best_tuple:
            best_tuple = t
            best_idx = idx

    for i, journey in enumerate(journeys, 1):
        path = journey["path"]
        costs = journey["costs"]
        trans = max(0, len(path) - 1)

        readable_path = [decode_trip(t) for t in path]
        path_text = " → ".join(readable_path)

        # Tag lines based on metrics
        tags = []
        if i - 1 == best_idx:
            tags.append("دي أسهل وأوفر رحلة ليك:")
        if trans == 0:
            tags.append("دي هتبقا مواصلة واحدة")
        elif trans > 1:
            tags.append("الرحلة دي محتاجة تغيير أكتر من ميكروباص:")
        if int(costs["walk"]) <= 10:
            tags.append("دي رحلة مفهاش مشي")
        if costs["money"] == max_money:
            tags.append("دي اغلي رحلة")
        if int(costs["walk"]) == int(min_walk):
            tags.append("دي أقل مشي")

        tag_text = "\n".join([f"🔸 {t}" for t in tags])
        tag_block = (tag_text + "\n") if tag_text else ""

        output += f"""
🔹 الرحلة {i}:
{tag_block}🛣 المسار: {path_text}
💰 السعر: {costs['money']} جنيه
🚶‍♂️ المشي: {int(costs['walk'])} متر
\n
"""

    return output

