from langchain.tools import tool
from typing import Dict, Any

from trip_decoder import decode_trip
from services.routing_client import find_route as grpc_find_route


def geocode_address(address: str) -> dict:
    """Geocode an address using the Alexandria-biased resolver.

    Implementation lives in `services/geocode.py` (DB-first then Nominatim fallback).
    """
    from services.geocode import geocode_address as svc_geo
    return svc_geo(address)
    
"""Tools for server-based routing.

Local-mode routing (OSMnx graph + explore_trips/find_journeys) is handled by the
backend routing server; the agent client should only geocode and call gRPC.
"""



@tool
def find_route_server(start_address: str, end_address: str, walking_cutoff: float = 5000.0, max_transfers: int = 2) -> Dict[str, Any]:
    """Geocode start/end, call gRPC FindRoute, and return journeys."""
    start = geocode_address(start_address)
    end = geocode_address(end_address)

    if "error" in start or "error" in end:
        return {"error": "تعذر تحديد المواقع. تأكد من العناوين."}

    result = grpc_find_route(
        start_lat=start["lat"],
        start_lon=start["lon"],
        end_lat=end["lat"],
        end_lon=end["lon"],
        walking_cutoff=walking_cutoff,
        max_transfers=max_transfers,
    )

    return result


@tool
def format_server_journeys_for_user(route_response: Dict[str, Any]) -> str:
    """Format gRPC route response into friendly Arabic guidance."""
    if not route_response or route_response.get("num_journeys", 0) == 0:
        return "لم يتم العثور على رحلات مناسبة بالقرب من نقطتي البداية أو النهاية ضمن مسافة المشي المحددة."

    journeys = route_response.get("journeys", [])

    output = ""
    for i, journey in enumerate(journeys, 1):
        path = journey.get("path", [])
        costs = journey.get("costs", {})

        readable_path = [decode_trip(t) for t in path]
        path_text = " → ".join(readable_path) if readable_path else "(مسار غير معروف)"

        money = costs.get("money", 0)
        walk_m = int(costs.get("walk", 0))
        time_min = int(costs.get("transport_time", 0))

        output += f"""
🔹 الرحلة {i}:
🛣 المسار: {path_text}
💰 السعر التقريبي: {money} جنيه
🚶‍♂️ إجمالي المشي: {walk_m} متر
⏱ زمن التنقل: ~{time_min} دقيقة

نصيحة: اتبع هذا التسلسل من الرحلات، وإذا احتجت مساعدة أثناء الطريق اسأل عن اسم الخط المذكور بين القوسين لكل مرحلة.
"""

    output += "\nنتمنى لك رحلة موفقة!"
    return output

