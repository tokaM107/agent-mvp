from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
from tools import geocode_address, find_route_server, format_server_journeys_for_user
from langchain.agents import create_agent
from services.geocode import geocode_address as svc_geocode
from services.routing_client import find_route as svc_find_route
from trip_decoder import decode_trip
import re, json
import google.generativeai as genai

system_prompt = """
أنت مساعد ذكي متخصص في مواصلات الإسكندرية.
استخدم الأدوات التالية بدقة:

1. geocode_address(address): للحصول على الإحداثيات.
2. find_route_server(start_address, end_address, walking_cutoff=1000, max_transfers=2): لاستدعاء خادم gRPC وإرجاع الرحلات.
3. format_server_journeys_for_user(route_response): لصياغة الرد النهائي بشكل عربي وودود.

التدفق المطلوب:
- فهم طلب المستخدم (نقطة بداية ونهاية).
- Geocode للعناوين.
- استدعاء السيرفر (FindRoute) للحصول على المسار.
- تحليل النتائج وصياغة رد نهائي واضح ولطيف يصلح للاستخدام العملي.
أعد فقط الرد النهائي للمستخدم.
"""

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=os.getenv("GOOGLE_API_KEY"))

tools = [geocode_address, find_route_server, format_server_journeys_for_user]

agent = create_agent(model, tools=tools)

def _llm_parse_places(query: str) -> tuple[str | None, str | None]:
    """Parse origin/destination via Google Generative AI directly (JSON-only)."""
    try:
        api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
        if not api_key:
            return None, None
        genai.configure(api_key=api_key)
        parse_prompt = (
            "أنت محلل نوايا. أخرج مكان الانطلاق والوصول من النص التالي وأعد JSON فقط بدون أي كلام إضافي،"
            " استعمل المفتاحين بالإنجليزية تمامًا: origin و destination. أعِدّ JSON مضغوط سطر واحد بدون أسطر جديدة ولا تعليقات"
            " وبدون أقواس أو تنسيق إضافي مثل ``` أو ```json. مثال دقيق: {\"origin\":\"الموقف الجديد\",\"destination\":\"العصافرة\"}.\n\n"
            f"النص: {query}"
        )
        resp = genai.GenerativeModel("gemini-2.5-flash").generate_content(parse_prompt, request_options={"retry": None, "timeout": 20})
        raw = getattr(resp, "text", "") or ""
        raw = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", raw.strip())
        raw = raw.strip()
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            return None, None
        data = json.loads(m.group(0))
        origin = (data.get("origin") or data.get("الانطلاق") or "").strip() or None
        dest = (data.get("destination") or data.get("الوصول") or "").strip() or None
        return origin, dest
    except Exception:
        return None, None


def run_once_server(query: str) -> str:
    """LLM-only parse, call gRPC server, format Arabic reply."""
    origin, dest = _llm_parse_places(query)
    if not origin or not dest:
        return (
            "تعذّر استخراج أماكن الانطلاق والوصول عبر النموذج. برجاء الكتابة بوضوح مثلاً: "
            "'أريد الذهاب من [المكان] إلى [المكان]'."
        )

    s = svc_geocode(origin)
    e = svc_geocode(dest)
    if "error" in s or "error" in e:
        return (
            "تعذّر تحديد المواقع. جرّب أسماء بديلة أو صيغة أدق.\n"
            f"Start: {origin} => {s}\nEnd  : {dest} => {e}"
        )

    resp = svc_find_route(
        start_lat=s["lat"], start_lon=s["lon"], end_lat=e["lat"], end_lon=e["lon"],
        walking_cutoff=5000.0, max_transfers=2,
    )
    
    # Inline formatting logic (instead of calling tool-wrapped function)
    formatted = _format_journeys(resp, origin, dest)

    # Try a final polish via LLM; fallback to raw formatted text
    try:
        polish = (
            f"المستخدم يريد الذهاب من {origin} إلى {dest}.\n\n"
            "من فضلك قدم نفس الرحلات التالية بشكل ودود وواضح باللهجة المصرية،"
            " دون تغيير الأسعار أو المسافات أو أسماء الخطوط:\n\n" + formatted
        )
        r = model.invoke(polish)
        return getattr(r, "content", str(r))
    except Exception:
        return formatted


def _format_journeys(route_response: dict, origin: str, dest: str) -> str:
    """Format gRPC route response into friendly Arabic guidance."""
    if not route_response or route_response.get("num_journeys", 0) == 0:
        return "لم يتم العثور على رحلات مناسبة بالقرب من نقطتي البداية أو النهاية ضمن مسافة المشي المحددة."

    journeys = route_response.get("journeys", [])

    output = f"**من {origin} إلى {dest}**\n\n"
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

"""

    output += "\nنصيحة: اتبع هذا التسلسل من الرحلات، وإذا احتجت مساعدة أثناء الطريق اسأل عن اسم الخط المذكور بين القوسين لكل مرحلة.\nنتمنى لك رحلة موفقة!"
    return output

# Add timeout to prevent hanging
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Agent call timed out after 60 seconds")

# Note: signal.alarm only works on Unix. For Windows, we'll catch KeyboardInterrupt
try:
    user_query = os.getenv("TEST_QUERY", "أريد الذهاب من الموقف الجديد إلى العصافرة")
    print("Starting agent... (Press Ctrl+C if it hangs)")
    # Prefer deterministic server path to avoid model quotas
    output = run_once_server(user_query)
    print("\n" + "="*60)
    print("Agent Response:")
    print("="*60)
    print(output)
except KeyboardInterrupt:
    print("\n\nInterrupted by user. Try running test_direct_call.py instead.")
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
    print("\nTry running: python test_direct_call.py")

# response = model.invoke("هو ازاي اروح من محطة مصر للعجمي ؟ في مشاريع بتروح هناك؟")
# print(response.content)