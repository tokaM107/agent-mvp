# graph_builder.py

import pandas as pd
import osmnx as ox

def attach_trips_to_graph(g):
    trips = pd.read_csv("data/trips.txt")
    routes = pd.read_csv("data/routes.txt")
    stop_times = pd.read_csv("data/stop_times.txt")
    stops = pd.read_csv("data/stops.txt")

    stop_to_trips = (
        stop_times.groupby('stop_id')['trip_id']
        .apply(list)
        .to_dict()
    )

    stop_nodes = ox.distance.nearest_nodes(
        g,
        X=stops['stop_lon'].values,
        Y=stops['stop_lat'].values
    )

    stop_to_node_map = pd.Series(stop_nodes, index=stops['stop_id']).to_dict()

    for stop_id, node_id in stop_to_node_map.items():
        trips_at_stop = stop_to_trips.get(stop_id)
        
        if trips_at_stop:
            if 'boarding_map' not in g.nodes[node_id]:
                g.nodes[node_id]['boarding_map'] = {}

            for trip_id in trips_at_stop:
                g.nodes[node_id]['boarding_map'][trip_id] = stop_id

    return g
