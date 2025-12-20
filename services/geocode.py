"""Simple geocode utility wrapper with Alexandria-focused fallbacks."""
from geopy.geocoders import Nominatim
from typing import List

try:
    # Optional import for better normalization
    from services.query_parser import SYNONYMS
except Exception:
    SYNONYMS = {}


def _alex_viewbox():
    # [min_lon, min_lat, max_lon, max_lat] roughly around Alexandria
    return [29.75, 31.10, 30.15, 31.35]


def _build_candidates(address: str) -> List[str]:
    base = address.strip()
    norm = SYNONYMS.get(base, base)
    candidates = []

    # Original and normalized
    candidates.append(base)
    if norm != base:
        candidates.append(norm)

    # Common transliterations and hints for known places
    lower = norm.lower()
    if "الموقف الجديد" in base or "mawqaf" in lower or "gedid" in lower or "gadid" in lower:
        candidates.extend([
            "New Bus Station",
            "Alexandria New Bus Station",
            "El Mawqaf El Gedid",
            "El Mawkaf El Gedid",
            "El Maw2af El Gedid",
            "Mawqaf El Gedid",
            "Maw2af El Gadid",
        ])
    elif "سيدي جابر" in base or "sidi" in lower and ("gaber" in lower or "gabir" in lower or "jaber" in lower):
        candidates.extend([
            "Sidi Gaber",
            "Sidi Gabir",
            "Sidi Jaber",
            "Sidi Gaber Station",
            "Sidi Gabir Station",
        ])
    elif "العصافرة" in base or "asafra" in lower or "asafera" in lower:
        candidates.extend([
            "Asafra",
            "Al Asafra",
            "Asafera",
        ])

    # Always include with city suffix
    extended = []
    for c in candidates:
        extended.append(c)
        extended.append(f"{c}, Alexandria, Egypt")
    # De-duplicate while preserving order
    seen = set()
    uniq = []
    for c in extended:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def geocode_address(address: str) -> dict:
    """Geocode an address to latitude/longitude, trying synonyms and viewbox bias."""
    geolocator = Nominatim(user_agent="alex_transit_agent")
    viewbox = _alex_viewbox()
    candidates = _build_candidates(address)

    # Try with Alexandria bounding first
    for q in candidates:
        try:
            loc = geolocator.geocode(
                q,
                exactly_one=True,
                country_codes="eg",
                addressdetails=False,
                timeout=10,
                viewbox=viewbox,
                bounded=True,
            )
        except Exception:
            loc = None
        if loc:
            return {"lat": float(loc.latitude), "lon": float(loc.longitude)}

    # Fallback: try without viewbox
    for q in candidates:
        try:
            loc = geolocator.geocode(
                q,
                exactly_one=True,
                country_codes="eg",
                addressdetails=False,
                timeout=10,
            )
        except Exception:
            loc = None
        if loc:
            return {"lat": float(loc.latitude), "lon": float(loc.longitude)}

    return {"error": "Location not found"}
