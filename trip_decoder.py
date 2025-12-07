import pandas as pd

trips_df = pd.read_csv("data/trips.txt")
routes_df = pd.read_csv("data/routes.txt")

def get_route_id_from_trip(trip_id):
    row = trips_df[trips_df["trip_id"] == trip_id].iloc[0]
    return row["route_id"]

def get_route_name(route_id):
    row = routes_df[routes_df["route_id"] == route_id].iloc[0]
    return {
        "long_name": row["route_long_name"],
        "short_name": row["route_short_name"],
        "type": row["route_type"]
    }

def decode_trip(trip_id):
    route_id = get_route_id_from_trip(trip_id)
    route_info = get_route_name(route_id)
    return f"{route_info['short_name']} ({route_info['long_name']})"
