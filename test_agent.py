from langchain.tools import tool
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

def run_agent(user_query: str):
    print(f"🔍 Analyzing: {user_query}")

    # 1. PARSE INTENT (Origin/Dest)
    parse_prompt = f"""
    Extract origin and destination from this Egyptian Arabic query.
    Return JSON only: {{"origin": "...", "destination": "..."}}
    Query: {user_query}
    """
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        resp = model.generate_content(parse_prompt)
        # Clean response to get pure JSON
        json_str = re.search(r"\{.*\}", resp.text, re.DOTALL).group(0)
        places = json.loads(json_str)
        origin_txt = places.get("origin")
        dest_txt = places.get("destination")
    except Exception as e:
        return f"❌ Error parsing query: {e}"

    if not origin_txt or not dest_txt:
        return "🤔 مش قادر افهم المكانين بالظبط، ممكن توضح اكتر؟"

    # 2. RESOLVE LOCATIONS (DB First, then Geocoding)
    # Origin
    src_candidates = search_stop_by_name_db(origin_txt)
    if src_candidates:
        src = src_candidates[0] # Best DB match
    else:
        geo = geocode_address(origin_txt)
        if "error" in geo: return f"📍 مش لاقي مكان اسمه '{origin_txt}'"
        src = get_nearest_stop_db(geo["lat"], geo["lon"])
        if not src: return "❌ مفيش محطات قريبة من نقطة البداية."

    # Destination
    dst_candidates = search_stop_by_name_db(dest_txt)
    if dst_candidates:
        dst = dst_candidates[0]
    else:
        geo = geocode_address(dest_txt)
        if "error" in geo: return f"📍 مش لاقي مكان اسمه '{dest_txt}'"
        dst = get_nearest_stop_db(geo["lat"], geo["lon"])
        if not dst: return "❌ مفيش محطات قريبة من نقطة الوصول."

    # 3. FIND JOURNEYS (DB + Logic)
    # Calculate Access/Egress Walking (Important!)
    # Note: src['distance_m'] is distance from User -> Stop
    #       dst['distance_m'] is distance from Stop -> Dest
    access_walk = src.get('distance_m', 0)
    egress_walk = dst.get('distance_m', 0)

    raw_journeys = find_journeys_db(src["stop_id"], dst["stop_id"])

    if not raw_journeys:
        return "🚫 للاسف مفيش مسارات مسجلة بين النقطتين دول في الداتا بيز حالياً."

    # 4. PREPARE DATA FOR GEMINI
    # Add access/egress walk to totals and decode names
    enhanced_journeys = []
    for j in raw_journeys:
        # Add the walk to the station and walk from station
        total_walk = j["costs"]["walk"] + access_walk + egress_walk
        
        # Decode path names
        readable_path = [decode_route_from_db(r) for r in j["path"]]
        
        enhanced_journeys.append({
            "path_names": readable_path,
            "total_price": j["costs"]["money"],
            "total_walk_meters": int(total_walk),
            "transfers": len(j["path"]) - 1
        })

    # 5. FINAL GEMINI RESPONSE (The "Brain")
    system_instruction = """
    You are a helpful Egyptian transportation assistant.
    You will receive a list of possible journeys (JSON).
    Your job is to pick the best ones and explain them to the user in friendly Egyptian Arabic.
    
    Analysis Rules:
    - Identify the CHEAPEST option.
    - Identify the LEAST WALKING option.
    - Identify the FASTEST/EASIEST option (usually fewer transfers).
    
    Response Format:
    - Start with a friendly greeting.
    - Suggest the best 1-2 options clearly.
    - Use emojis (🚌, 🚶, 💰).
    - Mention price and walking distance specifically (don't ignore them).
    - If walking is 0 or very low, highlight it as a "door-to-door" trip.
    """
    
    user_prompt = f"""
    User wants to go from: {origin_txt} to {dest_txt}.
    Here are the available options found in the database:
    {json.dumps(enhanced_journeys, ensure_ascii=False, indent=2)}
    
    Analyze these and give the best recommendation.
    """

    final_resp = model.generate_content(system_instruction + "\n" + user_prompt)
    return final_resp.text

if __name__ == "__main__":
    # Test
    print(run_agent("عايز اروح من سان استفانو للعصافرة"))