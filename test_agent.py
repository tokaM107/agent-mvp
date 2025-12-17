from langchain.tools import tool
import google.generativeai as genai
from geopy.geocoders import Nominatim
import json
import re
import os
import warnings
from pathlib import Path
from dotenv import load_dotenv


from tools import (
    search_stop_by_name_db,
    get_nearest_stop_db,
    find_journeys_db,
    filter_best_journeys,
    format_journeys_for_user,
    decode_trip,
    geocode_address
)


load_dotenv()
os.environ.setdefault("USE_DB", "1")       
os.environ.setdefault("DB_HOST", "localhost")
os.environ.setdefault("DB_NAME", "transport_db")
os.environ.setdefault("DB_USER", "postgres")
os.environ.setdefault("DB_PASSWORD", "postgres")

warnings.filterwarnings("ignore")

system_prompt = """
You are a smart assistant specialized in Alexandria public transportation.
Workflow:
1. Geocode origin/destination.
2. Find nearest stops in DB.
3. Find journeys using DB (SQL + ML Pricing).
4. Format output in Arabic.
"""

def run_once(query: str) -> str:
    # 1) (Origin / Destination)
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        return "⚠️ Error: GOOGLE_API_KEY is missing in .env file."
        
    genai.configure(api_key=api_key)
    
    parse_prompt = (
        "أنت محلل نوايا. أخرج مكان الانطلاق والوصول من النص التالي وأعد JSON فقط بدون أي كلام إضافي،"
        " استعمل المفتاحين بالإنجليزية تمامًا: origin و destination. "
        "مثال دقيق: {\"origin\":\"الموقف الجديد\",\"destination\":\"العصافرة\"}.\n\n"
        f"النص: {query}"
    )
    
    origin = None
    dest = None
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        parse_resp = model.generate_content(parse_prompt)
        raw = parse_resp.text.strip()
        # تنظيف الرد من علامات الكود
        raw = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", raw).strip()
        jmatch = re.search(r"\{[\s\S]*\}", raw)
        if jmatch:
            data = json.loads(jmatch.group(0))
            origin = data.get("origin") or data.get("الانطلاق")
            dest = data.get("destination") or data.get("الوصول")
    except Exception as e:
        print(f"[Error] parsing query: {e}")

    if not origin or not dest:
        return "عفواً، لم أستطع فهم مكان الانطلاق أو الوصول. يرجى التوضيح (مثال: من سموحة إلى المندرة)."

    print(f"\n📍 من: {origin} | 🏁 إلى: {dest}")

    # 2) البحث عن الإحداثيات وأقرب محطات (DB ONLY)
    # البحث بالاسم الأول
    found_src = search_stop_by_name_db(origin)
    found_dst = search_stop_by_name_db(dest)

    # تحديد إحداثيات البداية
    if found_src:
        src_geo = {"lat": found_src[0]["lat"], "lon": found_src[0]["lon"]}
        print(f"[DEBUG] Found Origin in DB: {found_src[0]['name']}")
    else:
        src_geo = geocode_address(origin) # Fallback to Nominatim if name not in DB

    # تحديد إحداثيات النهاية
    if found_dst:
        dst_geo = {"lat": found_dst[0]["lat"], "lon": found_dst[0]["lon"]}
        print(f"[DEBUG] Found Dest in DB: {found_dst[0]['name']}")
    else:
        dst_geo = geocode_address(dest)

    if "error" in src_geo or "error" in dst_geo:
        return "لم أتمكن من تحديد العناوين بدقة على الخريطة."

    # 3) تحديد أقرب محطة فعلية للإحداثيات (للتأكد وحساب مسافة المشي)
    db_near_src = get_nearest_stop_db(src_geo["lat"], src_geo["lon"])
    db_near_dst = get_nearest_stop_db(dst_geo["lat"], dst_geo["lon"])

    if not db_near_src or not db_near_dst:
        return "عفواً، لا توجد محطات مسجلة قريبة من هذه المواقع في قاعدة البيانات."

    print(f"[DEBUG] Nearest Stop (Start): {db_near_src['name']} ({int(db_near_src['distance_m'])}m walk)")
    print(f"[DEBUG] Nearest Stop (End):   {db_near_dst['name']} ({int(db_near_dst['distance_m'])}m walk)")
 
    journeys = find_journeys_db(
        origin_stop_id=db_near_src["stop_id"],
        dest_stop_id=db_near_dst["stop_id"],
        max_results=5,
        origin_walk_m=db_near_src["distance_m"],
        dest_walk_m=db_near_dst["distance_m"]
    )

    if not journeys:
        print("[DEBUG] find_journeys_db returned empty list.")
        return "للأسف، لم يتم العثور على مسارات مسجلة في قاعدة البيانات تربط بين هاتين المحطتين حالياً."

    
    best = filter_best_journeys(journeys, max_results=5)

    
    try:
        out_dir = Path("output")
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "origin": origin,
            "destination": dest,
            "journeys": best
        }
        with open(out_dir / "journeys.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    # (Gemini Polish)
    formatted_text = format_journeys_for_user(best)
    
    polish_prompt = (
        f"المستخدم يريد الذهاب من {origin} إلى {dest}.\n"
        "قم بصياغة الرد التالي بشكل جمالي باللهجة المصرية، مع الحفاظ على الأرقام والأسماء كما هي:\n\n"
        f"{formatted_text}"
    )
    
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        final_resp = model.generate_content(polish_prompt)
        return final_resp.text
    except Exception:
        return formatted_text

if __name__ == "__main__":
    user_query = "أريد الذهاب من الموقف الجديد الي العصافرة"
    print(f"🔹 السؤال: {user_query}")
    result = run_once(user_query)
    print("\n🔹 النتيجة النهائية:")
    print(result)