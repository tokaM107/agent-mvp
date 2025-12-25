from tools import geocode_address, format_server_journeys_for_user
from services.routing_client import find_route


coords_start = geocode_address("الموقف الجديد الاسكندرية")
coords_end = geocode_address("الاسكندرية العصافرة")

if "error" in coords_start or "error" in coords_end:
    raise SystemExit(f"Geocode failed: start={coords_start} end={coords_end}")

resp = find_route(
    start_lat=coords_start["lat"],
    start_lon=coords_start["lon"],
    end_lat=coords_end["lat"],
    end_lon=coords_end["lon"],
    walking_cutoff=1000.0,
    max_transfers=2,
)

print(format_server_journeys_for_user(resp))