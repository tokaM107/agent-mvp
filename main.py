from graph_builder import attach_trips_to_graph
import osmnx as ox
from models.trip_price_class import TripPricePredictor  

from tools import *
from services.pricing import get_cost
from services.distance import get_distance
from models.trip_price_class import load_model
import pandas as pd
from collections import defaultdict


g = ox.graph_from_xml("labeled.osm", bidirectional=True, simplify=True)
g = attach_trips_to_graph(g)
set_graph(g)

pathways = pd.read_csv('trip_pathways.csv')


trip_graph = defaultdict(dict)
pathways_dict = pathways.to_dict('index')

for idx, row in pathways.iterrows():
    trip_graph[row['start_trip_id']][row['end_trip_id']] = idx 
    
set_trip_graph(trip_graph, pathways_dict)

coords_start = geocode_address("الموقف الجديد الاسكندرية")
coords_end = geocode_address(" الاسكندرية العصافرة")
node_id_start = get_nearest_node(coords_start['lat'], coords_start['lon'])
node_id_end = get_nearest_node(coords_end['lat'], coords_end['lon'])
trips_start = explore_trips(node_id_start)
trips_end = explore_trips(node_id_end)
                            


# print(g)




jor = find_journeys(trip_graph, pathways_dict, trips_start, trips_end, max_transfers=2)
filtered= filter_best_journeys(jor, max_results=5)
formatted_journeys = format_journeys_for_user(filtered)
print(formatted_journeys)