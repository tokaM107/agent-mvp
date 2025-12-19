import google.generativeai as genai
from geopy.geocoders import Nominatim
import os
import re
import json
from dotenv import load_dotenv
from tools import *

load_dotenv()

# --- CONFIGURATION ---
API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    print("⚠️  WARNING: GOOGLE_API_KEY is not set!")

genai.configure(api_key=API_KEY)

# --- DICTIONARY ---
ARABIC_TO_ENGLISH = {
    'عصافرة': 'Asafra',
    'العصافرة': 'Asafra',
    'منشية': 'Mansheya',
    'المنشية': 'Mansheya',
    'منتزه': 'Montazah',
    'سيدي جابر': 'Sidi Gabir',
    'محطة الرمل': 'Raml Station',
    'رمل': 'Raml',
    'موقف جديد': 'Mawqaf Geded',
    'موقف': 'Mawqaf',
    'كيلو 21': 'Kilo 21',
    'الكيلو 21': 'Kilo 21',
    'محطة القطر': 'Train Station',
    'ميدان الشهداء': 'Shohada Square',
    'سان ستيفانو': 'San Stefano',
    'سان استفانو': 'San Stefano',
    'جليم': 'Gleem',
    'ستانلي': 'Stanley',
    'فيكتوريا': 'Victoria',
    'ميامي': 'Miami',
    'سيدي بشر': 'Sidi Bishr',
    'سموحة': 'Smouha',
    'السيوف': 'El Soyof',
    'العجمي': 'Agamy',
    'ابو قير': 'Abu Qir'
}

def normalize_arabic(text):
    """تنظيف النص العربي لضمان البحث في القاموس"""
    if not text: return ""
    text = text.strip()
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ة", "ه")
    # إزالة اللواصق في البداية (ل، ب، و، الـ) بما فيها التكرار زي "لل"
    while len(text) > 3 and (text.startswith("ال") or text[0] in ["ل", "ب", "و"]):
        if text.startswith("ال"):
            text = text[2:]
        else:
            text = text[1:]
    
    return text

def get_english_name(arabic_name):
    """ترجمة الاسم العربي للإنجليزي"""
    # 1. بحث مباشر
    if arabic_name in ARABIC_TO_ENGLISH: return ARABIC_TO_ENGLISH[arabic_name]
    
    # 2. بحث بعد التنظيف
    norm = normalize_arabic(arabic_name)
    for k, v in ARABIC_TO_ENGLISH.items():
        if normalize_arabic(k) == norm:
            return v
    
    # 3. لو مفيش، رجعه زي ما هو (للبحث الـ Fuzzy)
    return arabic_name 

def run_agent(user_query: str):
    print(f"🔍 Analyzing: {user_query}")

    # 1. GEMINI PARSING (Extraction Only)
    # جيميناي هنا دوره بس يطلع "المكان" من وسط كلام اليوزر
    parse_prompt = f"""
    You are a parser. Extract origin and destination from this Arabic query.
    Return strictly JSON: {{"origin": "...", "destination": "..."}}
    Rules:
    - Extract ONLY the location name (e.g., if "to El-Mansheya", return "El-Mansheya").
    - Do not translate to English yet. Keep it in Arabic if input is Arabic.
    Query: {user_query}
    """
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        resp = model.generate_content(parse_prompt)
        json_str = re.search(r"\{.*\}", resp.text, re.DOTALL).group(0)
        places = json.loads(json_str)
        origin_txt = places.get("origin")
        dest_txt = places.get("destination")
    except Exception as e:
        return f"❌ مش قادر افهم السؤال: {e}"

    if not origin_txt or not dest_txt:
        return "🤔 ياريت توضح المكانين (من ... إلى ...) عشان اقدر اساعدك."

    # 2. NORMALIZATION & TRANSLATION (Logic)
    origin_en = get_english_name(origin_txt)
    dest_en = get_english_name(dest_txt)
    
    print(f"[DEBUG] Origin: '{origin_txt}' → '{origin_en}'")
    print(f"[DEBUG] Destination: '{dest_txt}' → '{dest_en}'")

    # 3. DB RESOLUTION (Precision)
    # Origin
    src_candidates = search_stop_by_name_db(origin_en)
    if src_candidates:
        src = src_candidates[0]
        print(f"[DEBUG] Found Origin: {src['name']} (ID: {src['stop_id']})")
    else:
        geo = geocode_address(origin_en)
        if "error" in geo: return f"📍 مش لاقي مكان اسمه '{origin_txt}'"
        src = get_nearest_stop_db(geo["lat"], geo["lon"])
        if not src: return "❌ مفيش محطات قريبة من البداية."

    # Destination
    dst_candidates = search_stop_by_name_db(dest_en)
    if dst_candidates:
        dst = dst_candidates[0]
        print(f"[DEBUG] Found Destination: {dst['name']} (ID: {dst['stop_id']})")
    else:
        geo = geocode_address(dest_en)
        if "error" in geo: return f"📍 مش لاقي مكان اسمه '{dest_txt}'"
        dst = get_nearest_stop_db(geo["lat"], geo["lon"])
        if not dst: return "❌ مفيش محطات قريبة من الوصول."

    # 3.5 Always geocode both endpoints to compute access/egress walk
    # Prefer Arabic first, then English fallback
    o_geo = geocode_address(origin_txt)
    if "error" in o_geo:
        o_geo = geocode_address(origin_en)
    d_geo = geocode_address(dest_txt)
    if "error" in d_geo:
        d_geo = geocode_address(dest_en)

    # 4. FIND JOURNEYS
    # We'll compute access/egress per-option using each option's first/last stop ids.

    raw_journeys = find_journeys_db(src["stop_id"], dst["stop_id"])

    if not raw_journeys:
        return "🚫 للاسف مفيش مسارات مسجلة بين النقطتين دول حالياً."

    # 5. PREPARE DATA FOR GEMINI
    enhanced_journeys = []
    
    # Calculate min values for tagging
    all_prices = [j["costs"]["money"] for j in raw_journeys]
    # Temp placeholder; we will recompute per-option walks below
    all_walks = [j["costs"]["walk"] for j in raw_journeys]
    
    min_price = min(all_prices) if all_prices else 0
    min_walk = min(all_walks) if all_walks else 0
    
    for j in raw_journeys:
        # Per-option walk: Access (origin point -> first stop) + Transit Walk + Egress (last stop -> dest point)
        first_stop = j.get("stops_path", [src["stop_id"]])[0]
        last_stop = j.get("stops_path", [dst["stop_id"]])[-1]
        try:
            access_walk = 0 if "error" in o_geo else int(compute_walk_meters_point_to_stop(o_geo["lat"], o_geo["lon"], int(first_stop)) or 0)
            egress_walk = 0 if "error" in d_geo else int(compute_walk_meters_point_to_stop(d_geo["lat"], d_geo["lon"], int(last_stop)) or 0)
        except Exception:
            access_walk = int(src.get('distance_m', 0) or 0)
            egress_walk = int(dst.get('distance_m', 0) or 0)
        total_walk = j["costs"]["walk"] + access_walk + egress_walk
        total_price = j["costs"]["money"]
        transfers = max(0, len(j["path"]) - 1)
        
        tags = []
        if total_price <= min_price: tags.append("الأوفر 💰")
        if total_walk <= min_walk + 50: tags.append("أقل مشي 🚶")
        if transfers == 0: tags.append("مباشر 🚌")
        elif transfers == 1: tags.append("تبديلة واحدة")
        
        readable_path = [decode_route_from_db(r) for r in j["path"]]
        
        enhanced_journeys.append({
            "routes": readable_path,
            "price": total_price,
            "walk_meters": int(total_walk),
            "access_walk_m": int(access_walk),
            "egress_walk_m": int(egress_walk),
            "transfers": transfers,
            "tags": " - ".join(tags) if tags else "رحلة عادية"
        })

    # 6.1 Add pure walking option (network if available, else geodesic)
    try:
        if "error" not in o_geo and "error" not in d_geo:
            walk_only_m = compute_walk_meters_point_to_point(o_geo["lat"], o_geo["lon"], d_geo["lat"], d_geo["lon"]) or 0
            enhanced_journeys.append({
                "routes": ["مشي"],
                "price": 0.0,
                "walk_meters": int(walk_only_m),
                "access_walk_m": int(walk_only_m),
                "egress_walk_m": 0,
                "transfers": 0,
                "tags": "مشي فقط"
            })
    except Exception:
        pass

    # 6. FINAL GEMINI RESPONSE
    system_instruction = """
    أنت خبير مواصلات إسكندراني.
    مهمتك: صياغة الرد النهائي لليوزر بناءً على البيانات المقدمة فقط.
    
    القواعد:
    1. اتكلم بلهجة مصرية ودودة.
    2. اعرض الخيارات بوضوح (الخيار الأول، الثاني..).
    3. ركز على "الوصف" (ده الأوفر، ده الأسرع..).
    4. اشرح المسار: "هتمشي {access_walk_m} متر وتاخد كذا.. وتنزل تمشي {egress_walk_m} متر".
    5. اكتب السعر والمشي بدقة من البيانات.
    6. لو المسار مشي فقط، قول "المسافة قريبة، تمشاها أحسن".
    """
    
    user_data = f"""
    سؤال اليوزر: من {origin_txt} إلى {dest_txt}
    البيانات (JSON): {json.dumps(enhanced_journeys, ensure_ascii=False)}
    """

    try:
        final_resp = model.generate_content(system_instruction + "\n" + user_data)
        return final_resp.text
    except:
        return str(enhanced_journeys)

if __name__ == "__main__":
    q = "عايز اروح من العصافرة للكيلو 21 "
    print(run_agent(q))