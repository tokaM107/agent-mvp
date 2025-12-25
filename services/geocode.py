"""Hybrid geocode utility: DB stops first (pg_trgm), then Nominatim.

Goal:
- Prefer authoritative DB stop locations to avoid Nominatim ambiguity.
- Fallback to Nominatim only when DB match quality is low.

Important:
- DB stores geometry in PostGIS; when constructing points use ST_Point(lon, lat).
"""

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import os


def _normalize_ar(text: str) -> str:
    # Minimal normalization for common Arabic spelling variants.
    # Example: ي vs ى
    return (text or "").replace("ى", "ي").strip()


def _search_stop_db(query: str) -> dict | None:
    """Return best stop match from Postgres using pg_trgm similarity.

    Returns: {lat, lon, score, stop_id, name} or None
    """
    try:
        import psycopg2
    except Exception:
        return None

    host = os.environ.get("DB_HOST", "localhost")
    port = int(os.environ.get("DB_PORT", "5432"))
    db = os.environ.get("DB_NAME", "transport_db")
    user = os.environ.get("DB_USER", "postgres")
    pwd = os.environ.get("DB_PASSWORD", "postgres")

    q_norm = _normalize_ar(query)
    if not q_norm:
        return None

    threshold = float(os.environ.get("STOP_SIM_THRESHOLD", "0.22"))

    try:
        conn = psycopg2.connect(host=host, port=port, database=db, user=user, password=pwd)
    except Exception:
        return None

    try:
        cur = conn.cursor()
        # Apply simple normalization in SQL too so ي/ى variants compare well.
        sql = (
            "SELECT stop_id, name, ST_X(geom_4326) AS lon, ST_Y(geom_4326) AS lat, "
            "       similarity(translate(name,'ى','ي'), %s) AS score "
            "FROM stop "
            "WHERE translate(name,'ى','ي') % %s "
            "ORDER BY score DESC, name ASC "
            "LIMIT 1;"
        )
        cur.execute(sql, (q_norm, q_norm))
        row = cur.fetchone()
        if not row:
            return None
        stop_id, name, lon, lat, score = row
        if score is None or float(score) < threshold:
            return None
        return {
            "stop_id": int(stop_id),
            "name": str(name),
            "lon": float(lon),
            "lat": float(lat),
            "score": float(score),
        }
    except Exception:
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass


def geocode_address(address: str) -> dict:
    """Geocode an address to latitude/longitude.

    1) Try Postgres `stop` table with pg_trgm similarity (best effort).
    2) Fallback to Nominatim with Alexandria bias and 1s RateLimiter.
    """

    # 1) DB stops lookup
    db_hit = _search_stop_db(address)
    if db_hit:
        return {"lat": db_hit["lat"], "lon": db_hit["lon"]}

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
