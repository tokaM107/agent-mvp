from langchain.tools import tool
import google.generativeai as genai
from geopy.geocoders import Nominatim
import time
import pandas as pd
from models.trip_price_class import TripPricePredictor
from tools import *
from dotenv import load_dotenv
from pathlib import Path
import os
import re
import json

# Load environment variables
load_dotenv()

print(" DB-only mode: skipping OSM graph initialization")


print(" DB-only mode: skipping pathways graph caching")




system_prompt = """
You are a smart assistant specialized in Alexandria public transportation.
You must use DATABASE tools only:

1. search_stop_by_name_db(name) -> returns candidate stops from DB.
2. get_nearest_stop_db(lat, lon) -> returns nearest DB stop.
3. find_journeys_db(origin_stop_id, dest_stop_id) -> returns journeys (path, money, walk).
4. filter_best_journeys(journeys) -> filter top journeys.
5. format_journeys_for_user(journeys) -> Arabic formatted output.

Workflow:
1. Parse origin/destination via Gemini.
2. Resolve to DB stops and compute journeys via find_journeys_db.
3. Return formatted journeys only.

Output style requirements:
- Be clear, friendly, and concise in Arabic.
- Use headings, bullets, and icons (🛣 💰 🚶‍♂️).
- Avoid raw JSON; return human-friendly text only.
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
    # Prefer DB stop name search first; else use geocoder
    db_first = os.environ.get("USE_DB", "").strip()
    found_src = search_stop_by_name_db(origin) if db_first else []
    found_dst = search_stop_by_name_db(dest) if db_first else []
    if found_src:
        src_geo = {"lat": found_src[0]["lat"], "lon": found_src[0]["lon"], "db_stop_id": found_src[0]["stop_id"]}
    else:
        src_geo = geocode_address(origin)
    if found_dst:
        dst_geo = {"lat": found_dst[0]["lat"], "lon": found_dst[0]["lon"], "db_stop_id": found_dst[0]["stop_id"]}
    else:
        dst_geo = geocode_address(dest)
    if "error" in src_geo or "error" in dst_geo:
        return "لم أستطع تحديد العناوين بدقة. جرّب صيغة أخرى." 
    # Try DB nearest stop to verify correctness; fallback to OSM node
    db_near_src = get_nearest_stop_db(src_geo["lat"], src_geo["lon"]) if db_first else None
    db_near_dst = get_nearest_stop_db(dst_geo["lat"], dst_geo["lon"]) if db_first else None
    src_node = get_nearest_node(src_geo["lat"], src_geo["lon"]) 
    dst_node = get_nearest_node(dst_geo["lat"], dst_geo["lon"]) 

    # Optional debug: show resolved coordinates and node ids
    if os.environ.get("DEBUG_ROUTING", "").strip():
        print(f"[DEBUG] origin='{origin}' -> lat={src_geo['lat']}, lon={src_geo['lon']}, node={src_node}, db_nearest={db_near_src}")
        print(f"[DEBUG] dest='{dest}' -> lat={dst_geo['lat']}, lon={dst_geo['lon']}, node={dst_node}, db_nearest={db_near_dst}")
    # Prefer DB journeys if DB is enabled and nearest DB stops are available
    best = []
    if db_first and db_near_src and db_near_dst:
        db_journeys = find_journeys_db(db_near_src["stop_id"], db_near_dst["stop_id"], max_results=5)
        if db_journeys:
            best = filter_best_journeys(db_journeys, max_results=5)

    # NO FALLBACK: Database only mode
    if not best:
        return " لم يتم العثور على مسارات متاحة في قاعدة البيانات. تأكد من أن المحطات موجودة في النظام."

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
                    "decoded_path": [decode_trip(t) if isinstance(t, (int, float)) or str(t).isdigit() else str(t) for t in j["path"]],
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