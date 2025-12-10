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
import os
import pickle

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


def run_once(query: str) -> str:
    # 1) LLM parses the user query to origin/destination once (small prompt)
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    genai.configure(api_key=api_key)
    parse_prompt = (
        "استخرج مكان الانطلاق والوصول من الجملة التالية بدقة، بصيغة عربية بسيطة فقط كالتالي:\n"
        "الانطلاق: ...\nالوصول: ...\n\nالجملة:\n" + query
    )
    try:
        parse_resp = genai.GenerativeModel("gemini-pro").generate_content(parse_prompt, request_options={"retry": None, "timeout": 30})
        parse_text = getattr(parse_resp, "text", "")
    except Exception:
        parse_text = "الانطلاق: الموقف الجديد\nالوصول: العصافرة"

    # naive extract
    origin = "الموقف الجديد"
    dest = "العصافرة"
    for line in parse_text.splitlines():
        if line.strip().startswith("الانطلاق:"):
            origin = line.split(":", 1)[-1].strip() or origin
        if line.strip().startswith("الوصول:"):
            dest = line.split(":", 1)[-1].strip() or dest

    # 2) Tools pipeline (deterministic)
    src_geo = geocode_address(origin)
    dst_geo = geocode_address(dest)
    if "error" in src_geo or "error" in dst_geo:
        return "لم أستطع تحديد العناوين بدقة. جرّب صيغة أخرى." 
    src_node = get_nearest_node(src_geo["lat"], src_geo["lon"]) 
    dst_node = get_nearest_node(dst_geo["lat"], dst_geo["lon"]) 
    start_trips = explore_trips(src_node)
    goal_trips = explore_trips(dst_node)
    journeys = find_journeys(start_trips, goal_trips)
    best = filter_best_journeys(journeys, max_results=5)
    formatted = format_journeys_for_user(best)

    # 3) Single LLM call for final Arabic answer, prefer 2.5-flash then fallback to pro
    polish_prompt = (
        "أكد للمستخدم المسار المقترح التالي بشكل طبيعي ومفهوم، واستخدم لهجته المصرية إن أمكن،"
        " مع الحفاظ على الأسعار والمسافات والأسماء كما هي تمامًا.\n\n" + formatted
    )
    try:
        resp = genai.GenerativeModel("gemini-2.5-flash").generate_content(polish_prompt, request_options={"retry": None, "timeout": 60})
        return getattr(resp, "text", str(resp))
    except Exception:
        try:
            resp = genai.GenerativeModel("gemini-pro").generate_content(polish_prompt, request_options={"retry": None, "timeout": 60})
            return getattr(resp, "text", str(resp))
        except Exception:
            return formatted

if __name__ == "__main__":
    user_query = "أريد الذهاب من محطة مصر  الي ابو يوسف"
    print("🚀 السؤال:", user_query)
    out = run_once(user_query)
    print("\n🏁 النتيجة النهائية:")
    print(out)
# agent.run("أريد الذهاب من العجمي إلى محطة الرمل")

# response = model.invoke("هو ازاي اروح من محطة مصر للعجمي ؟ في مشاريع بتروح هناك؟")
# print(response.content)