import sys
from services.query_parser import parse_arabic_route_query, normalize_name
from services.geocode import geocode_address
from services.routing_client import find_route
from tools import format_server_journeys_for_user


def main():
    if len(sys.argv) >= 2:
        query = " ".join(sys.argv[1:])
    else:
        query = input("أدخل سؤالك (مثال: أريد الذهاب من الموقف الجديد إلى العصافرة):\n> ")

    start, end = parse_arabic_route_query(query)
    if not start or not end:
        print("لم أفهم صيغة الطلب. برجاء الكتابة بهذا الشكل: 'أريد الذهاب من X إلى Y'")
        return

    start_norm = normalize_name(start)
    end_norm = normalize_name(end)

    s = geocode_address(start_norm)
    e = geocode_address(end_norm)
    if "error" in s or "error" in e:
        print("تعذر تحديد المواقع. تأكد من الأسماء أو جرّب أسماء بديلة.")
        print(f"Start: {start_norm} => {s}")
        print(f"End  : {end_norm} => {e}")
        return

    # Call server directly with coordinates
    resp = find_route(
        start_lat=s["lat"],
        start_lon=s["lon"],
        end_lat=e["lat"],
        end_lon=e["lon"],
        walking_cutoff=2000.0,
        max_transfers=2,
    )

    # Format
    text = format_server_journeys_for_user(resp)
    print("\n" + "="*60)
    print("الرحلات المقترحة:")
    print("="*60)
    print(text)


if __name__ == "__main__":
    main()
