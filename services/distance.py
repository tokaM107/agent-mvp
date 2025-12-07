import pandas as pd

def get_distance(trip_id, start_stop, end_stop):
    # query db here, for now return total distance from trip_distances.csv
    dist = pd.read_csv("trip_distances.csv")
    return dist[dist["trip_id"] == trip_id]["distance_km"].values[0]

