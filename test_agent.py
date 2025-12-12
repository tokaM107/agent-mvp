from langchain.tools import tool
import google.generativeai as genai
from geopy.geocoders import Nominatim
import time
import pandas as pd
import osmnx as ox
from models.trip_price_class import TripPricePredictor
import heapq
from tools import *
from dotenv import load_dotenv
from pathlib import Path
import os
import pickle
import re
import json

# Load environment variables
load_dotenv()

# Cache the graph to avoid reprocessing each run
CACHE_GRAPH_PATH = "graph_cache.pkl"
if os.path.exists(CACHE_GRAPH_PATH):
    with open(CACHE_GRAPH_PATH, "rb") as f:
        g = pickle.load(f)
else:
    g = ox.graph_from_xml("labeled.osm", bidirectional=True, simplify=True)
    g = attach_trips_to_graph(g)
    with open(CACHE_GRAPH_PATH, "wb") as f:
        pickle.dump(g, f)
set_graph(g)

print("✅ Graph initialized")
print(g.nodes[list(g.nodes)[0]].keys())


# Cache pathways graph mapping
CACHE_PATHWAYS_PATH = "pathways_cache.pkl"
if os.path.exists(CACHE_PATHWAYS_PATH):
    with open(CACHE_PATHWAYS_PATH, "rb") as f:
        trip_graph, pathways_dict = pickle.load(f)
else:
    pathways = pd.read_csv('trip_pathways.csv')
    trip_graph = defaultdict(dict)
    pathways_dict = pathways.to_dict('index')
    for idx, row in pathways.iterrows():
        trip_graph[row['start_trip_id']][row['end_trip_id']] = idx 
    with open(CACHE_PATHWAYS_PATH, "wb") as f:
        pickle.dump((trip_graph, pathways_dict), f)
set_trip_graph(trip_graph, pathways_dict)




system_prompt = """
You are a smart assistant specialized in Alexandria public transportation. 
You have access to the following tools:

1. geocode_address(address) -> returns the latitude and longitude of the address.
2. get_nearest_node(lat, lon) -> returns the nearest OSM node ID.
3. explore_trips(source_node) -> returns all trips starting from this node, including walking distance.
4. find_journeys(start_trips, goal_trips) -> returns all possible journeys with path and costs (money, walking distance).
5. filter_best_journeys(journeys, max_results=5) -> returns the best journeys based on shortest walking distance and lowest cost.
6. format_journeys_for_user(journeys) -> returns a user-friendly Arabic description of the journeys.

You must always follow this workflow:
1. Find the coordinates of the start and destination using geocode_address. IMPORTANT: Always append ", Alexandria, Egypt" to the address provided by the user to ensure accuracy (e.g., if user says "Asafra", search for "Asafra, Alexandria, Egypt").
2. Convert each location into the nearest OSM node using get_nearest_node.
3. Explore trips from both start and destination nodes using explore_trips.
4. Find all possible journeys using find_journeys.
5. Filter the top journeys using filter_best_journeys.
6. Format the filtered journeys for the user using format_journeys_for_user.
7. Return only the final formatted journey description to the user. Do not return any intermediate data.

Output style requirements:
- Be clear, friendly, and concise in Arabic.
- Use headings, bullets, and icons (🛣 💰 🚶‍♂️) similar to the tools output.
- Start with a brief confirmation of origin and destination, then list top journeys.
- For each journey: show the path (trip names), total price, and total walking distance.
- Avoid raw JSON; return a human-friendly formatted text only.
"""


def _regex_extract(query: str) -> tuple[str, str] | tuple[None, None]:
    # Disabled: fallback extraction via regex is no longer used.
    return None, None


def run_once(query: str) -> str:
    # 1) LLM parses the user query to origin/destination once (JSON-only)
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    genai.configure(api_key=api_key)
    parse_prompt = (
        "أنت محلل نوايا. أخرج مكان الانطلاق والوصول من النص التالي وأعد JSON فقط بدون أي كلام إضافي،"
        " استعمل المفتاحين بالإنجليزية تمامًا: origin و destination. أعِدّ JSON مضغوط سطر واحد بدون أسطر جديدة ولا تعليقات"
        " وبدون أقواس أو تنسيق إضافي مثل ``` أو ```json. مثال دقيق: {\"origin\":\"الموقف الجديد\",\"destination\":\"العصافرة\"}.\n\n"
        f"النص: {query}"
    )
    origin = None
    dest = None
    try:
        # Use a modern fast model for parsing
        parse_resp = genai.GenerativeModel("gemini-2.5-flash").generate_content(parse_prompt, request_options={"retry": None, "timeout": 20})
        raw = getattr(parse_resp, "text", "") or ""
        # Strip common code fences/backticks and language tags
        raw = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", raw.strip())
        raw = raw.strip()
        # Extract first JSON object from text
        jmatch = re.search(r"\{[\s\S]*\}", raw)
        if jmatch:
            data = json.loads(jmatch.group(0))
            origin = (data.get("origin") or data.get("الانطلاق") or "").strip() or None
            dest = (data.get("destination") or data.get("الوصول") or "").strip() or None
    except Exception:
        pass

    # Require valid LLM parse; no regex fallback
    if not origin or not dest:
        return "تعذّر استخراج أماكن الانطلاق والوصول عبر Gemini. تأكّد من كتابة الصيغة بوضوح (مثال: من [المكان A] إلى [المكان B]) وبوجود مفتاح API صالح."

    # 2) Tools pipeline (deterministic)
    src_geo = geocode_address(origin)
    dst_geo = geocode_address(dest)
    if "error" in src_geo or "error" in dst_geo:
        return "لم أستطع تحديد العناوين بدقة. جرّب صيغة أخرى." 
    src_node = get_nearest_node(src_geo["lat"], src_geo["lon"]) 
    dst_node = get_nearest_node(dst_geo["lat"], dst_geo["lon"]) 

    # Optional debug: show resolved coordinates and node ids
    if os.environ.get("DEBUG_ROUTING", "").strip():
        print(f"[DEBUG] origin='{origin}' -> lat={src_geo['lat']}, lon={src_geo['lon']}, node={src_node}")
        print(f"[DEBUG] dest='{dest}' -> lat={dst_geo['lat']}, lon={dst_geo['lon']}, node={dst_node}")
    start_trips = explore_trips(src_node)
    goal_trips = explore_trips(dst_node)
    journeys = find_journeys(start_trips, goal_trips)
    best = filter_best_journeys(journeys, max_results=5)

    # Persist journeys to JSON for later querying
    try:
        out_dir = Path("output"); out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "origin": origin,
            "destination": dest,
            "origin_geo": src_geo,
            "destination_geo": dst_geo,
            "start_node": src_node,
            "dest_node": dst_node,
            "journeys": [
                {
                    "path": j["path"],
                    "decoded_path": [decode_trip(t) for t in j["path"]],
                    "costs": j["costs"],
                    "transfers": max(0, len(j["path"]) - 1)
                } for j in best
            ]
        }
        with open(out_dir / "journeys.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    formatted = format_journeys_for_user(best)

    # 3) Single LLM call for final Arabic answer, prefer 2.5-flash then fallback to pro
    polish_prompt = (
        f"المستخدم يريد الذهاب من {origin} إلى {dest}.\n\n" 
        "أكد للمستخدم المسار المقترح التالي بشكل طبيعي ومفهوم، واستخدم لهجته المصرية إن أمكن،"
        " مع الحفاظ على الأسعار والمسافات والأسماء كما هي تمامًا.\n\n" + formatted
    )
    try:
        resp = genai.GenerativeModel("gemini-2.5-flash").generate_content(polish_prompt, request_options={"retry": None, "timeout": 60})
        return getattr(resp, "text", str(resp))
    except Exception:
        try:
            resp = genai.GenerativeModel("gemini-2.5-flash").generate_content(polish_prompt, request_options={"retry": None, "timeout": 60})
            return getattr(resp, "text", str(resp))
        except Exception:
            return formatted


def query_saved_journeys() -> dict | None:
    """Load last saved journeys from output/journeys.json."""
    p = Path("output/journeys.json")
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

if __name__ == "__main__":
    user_query = "أريد الذهاب من الموقف الجديد الي العصافرة"
    print(" السؤال:", user_query)
    out = run_once(user_query)
    print(" النتيجة النهائية:")
    print(out)
    saved = query_saved_journeys()
    if saved:
        print("\n[Saved] output/journeys.json written with current results.")
# agent.run("أريد الذهاب من العجمي إلى محطة الرمل")

# response = model.invoke("هو ازاي اروح من محطة مصر للعجمي ؟ في مشاريع بتروح هناك؟")
# print(response.content)