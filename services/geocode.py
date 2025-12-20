"""Hybrid geocode utility: try local GTFS stops first, then Nominatim.

Respects Nominatim rate limits by using a single request with Alexandria bias.
"""
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from typing import Optional
import pandas as pd
import os


# Load local stops dataset once (GTFS)
_STOPS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "stops.txt")
_STOPS_DF = None
if os.path.exists(_STOPS_PATH):
    try:
        _STOPS_DF = pd.read_csv(_STOPS_PATH)
        # Prepare lowercase column for contains search
        if "stop_name" in _STOPS_DF.columns:
            _STOPS_DF["_stop_name_lc"] = _STOPS_DF["stop_name"].astype(str).str.lower()
    except Exception:
        _STOPS_DF = None


def _search_local_stop(address: str) -> Optional[dict]:
    if _STOPS_DF is None:
        return None
    name = address.strip().lower()
    # First try exact-ish contains match on stop_name
    match = _STOPS_DF[_STOPS_DF["_stop_name_lc"].str.contains(name, na=False)]
    if not match.empty:
        row = match.iloc[0]
        try:
            return {"lat": float(row["stop_lat"]), "lon": float(row["stop_lon"]), "source": "local"}
        except Exception:
            return None
    return None


def geocode_address(address: str) -> dict:
    """Geocode an address to latitude/longitude.

    1) Try local GTFS stops (data/stops.txt) for fast, precise station lookup.
    2) Fallback to Nominatim with Alexandria bias and 1s RateLimiter.
    """
    # 1) Local stops lookup
    local = _search_local_stop(address)
    if local:
        return {"lat": local["lat"], "lon": local["lon"]}

    # 2) Nominatim fallback (single query, Alexandria bias)
    geolocator = Nominatim(user_agent="alex_transit_agent")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.0)
    query = address.strip()
    if ("Alexandria" not in query) and ("الإسكندرية" not in query):
        query = f"{query}, Alexandria, Egypt"

    try:
        location = geocode(query, exactly_one=True, country_codes="eg", addressdetails=False, timeout=10)
    except Exception:
        location = None

    if location:
        return {"lat": float(location.latitude), "lon": float(location.longitude)}
    else:
        return {"error": "Location not found"}
