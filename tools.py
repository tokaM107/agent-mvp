from langchain.tools import tool
from geopy.geocoders import Nominatim
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


# @tool
def geocode_address(address: str) -> dict:
    """Geocode an address to latitude and longitude coordinates.
    Always bias search to Alexandria, Egypt for accuracy.
    """
    geolocator = Nominatim(user_agent="alex-transport-agent")
    # Ensure locality context for better accuracy
    query = f"{address}, Alexandria, Egypt" if "Alexandria" not in address and "الإسكندرية" not in address else address
    location = geolocator.geocode(query)

    if location:
        return {
            "lat": float(location.latitude),
            "lon": float(location.longitude)
        }
    else:
        return {
            "error": "Location not found"
        }
    
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
                        "stop_id": real_stop_id, # SAVE THE GTFS STOP ID
                        "osm_node_id": node,        # Save OSM ID just in case
                        "walk": d,
                        "path": reconstruct_path(node)
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

    for i, journey in enumerate(journeys, 1):
        path = journey["path"]
        costs = journey["costs"]

        readable_path = [decode_trip(t) for t in path]
        path_text = " → ".join(readable_path)

        output += f"""
🔹 الرحلة {i}:
🛣 المسار: {path_text}
💰 السعر: {costs['money']} جنيه
🚶‍♂️ المشي: {int(costs['walk'])} متر
\n
"""

    return output

