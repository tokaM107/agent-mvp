from typing import TypedDict, Optional, Dict, Any, List
from dataclasses import dataclass
import os
import re
import json
from dotenv import load_dotenv

# LangGraph
from langgraph.graph import StateGraph, END

# Services
from services.geocode import geocode_address as svc_geocode
from services.routing_client import find_route as svc_find_route
from trip_decoder import decode_trip

# LLMs (new preferred google.genai, fallback to deprecated google.generativeai)
try:
    from google.genai import Client as GenAIClient
    _GENAI_NEW = True
except Exception:
    _GENAI_NEW = False
    import google.generativeai as genai


class AgentState(TypedDict):
    query: str
    origin: Optional[str]
    destination: Optional[str]
    origin_geo: Optional[Dict[str, float]]
    destination_geo: Optional[Dict[str, float]]
    route_response: Optional[Dict[str, Any]]
    formatted: Optional[str]


def _clean_json_text(text: str) -> str:
    # Strip code fences and language tags
    txt = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", text.strip())
    # Extract first JSON object
    m = re.search(r"\{[\s\S]*\}", txt)
    return m.group(0) if m else ""


def parse_query_with_llm(query: str) -> tuple[Optional[str], Optional[str]]:
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        return None, None

    try:
        if _GENAI_NEW:
            client = GenAIClient(api_key=api_key)
            resp = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[{"role": "user", "parts": [
                    (
                        "أنت محلل نوايا. أخرج مكان الانطلاق والوصول من النص التالي وأعد JSON فقط بدون أي كلام إضافي، "
                        "استعمل المفتاحين بالإنجليزية تمامًا: origin و destination. أعِدّ JSON مضغوط سطر واحد بدون أسطر جديدة ولا تعليقات "
                        "ودون أي أقواس أو تنسيق إضافي مثل ``` أو ```json. مثال: {\"origin\":\"الموقف الجديد\",\"destination\":\"العصافرة\"}.\n\n"
                        f"النص: {query}"
                    )
                ]}],
                response_mime_type="application/json",
                # 20s timeout equivalent not exposed; rely on client defaults
            )
            raw = getattr(resp, "text", None)
            if raw is None:
                # Some versions return candidates
                raw = resp.candidates[0].content.parts[0].text if getattr(resp, "candidates", None) else ""
        else:
            genai.configure(api_key=api_key)
            resp = genai.GenerativeModel("gemini-2.5-flash").generate_content(
                (
                    "أنت محلل نوايا. أخرج مكان الانطلاق والوصول من النص التالي وأعد JSON فقط بدون أي كلام إضافي، "
                    "استعمل المفتاحين بالإنجليزية تمامًا: origin و destination. أعِدّ JSON مضغوط سطر واحد بدون أسطر جديدة ولا تعليقات "
                    "ودون أي أقواس أو تنسيق إضافي مثل ``` أو ```json. مثال: {\"origin\":\"الموقف الجديد\",\"destination\":\"العصافرة\"}.\n\n"
                    f"النص: {query}"
                ),
                request_options={"retry": None, "timeout": 20}
            )
            raw = getattr(resp, "text", "") or ""

        cleaned = _clean_json_text(raw)
        data = json.loads(cleaned) if cleaned else {}
        origin = (data.get("origin") or data.get("الانطلاق") or "").strip() or None
        dest = (data.get("destination") or data.get("الوصول") or "").strip() or None
        return origin, dest
    except Exception:
        return None, None


_route_re = re.compile(r"من\s+(.*?)\s+إلى\s+(.*)")

def parse_query_fallback_regex(query: str) -> tuple[Optional[str], Optional[str]]:
    m = _route_re.search(query)
    if not m:
        return None, None
    origin = m.group(1).strip()
    dest = m.group(2).strip()
    return origin or None, dest or None


def node_parse(state: AgentState) -> AgentState:
    origin, dest = parse_query_with_llm(state["query"])
    if not origin or not dest:
        # Fallback to deterministic regex to ensure end-to-end result
        origin, dest = parse_query_fallback_regex(state["query"])
        if not origin or not dest:
            state["formatted"] = (
                "تعذّر استخراج أماكن الانطلاق والوصول عبر النموذج. برجاء الكتابة بوضوح مثلاً: 'أريد الذهاب من [المكان] إلى [المكان]'."
            )
            return state
    state["origin"] = origin
    state["destination"] = dest
    return state


def node_geocode(state: AgentState) -> AgentState:
    if not state.get("origin") or not state.get("destination"):
        return state
    s = svc_geocode(state["origin"])  # hybrid: DB stops first, then Nominatim
    e = svc_geocode(state["destination"])  # hybrid
    state["origin_geo"] = None if "error" in s else {"lat": s["lat"], "lon": s["lon"]}
    state["destination_geo"] = None if "error" in e else {"lat": e["lat"], "lon": e["lon"]}
    if state["origin_geo"] is None or state["destination_geo"] is None:
        state["formatted"] = (
            "تعذّر تحديد المواقع. جرّب أسماء بديلة أو صيغة أدق.\n"
            f"Start: {state['origin']} => {s}\nEnd  : {state['destination']} => {e}"
        )
    return state


def node_route(state: AgentState) -> AgentState:
    if not state.get("origin_geo") or not state.get("destination_geo"):
        return state
    s = state["origin_geo"]; e = state["destination_geo"]
    resp = svc_find_route(
        start_lat=s["lat"], start_lon=s["lon"], end_lat=e["lat"], end_lon=e["lon"],
        walking_cutoff=1000.0, max_transfers=2,
    )
    state["route_response"] = resp
    return state


def _gemini_polish(text: str) -> str:
    """Polish the response with a friendly Egyptian tone using Gemini"""
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return text
        
        # Try new google.genai first
        try:
            import google.genai as genai
            genai.configure(api_key=api_key)
            resp = genai.GenerativeModel("gemini-1.5-flash").generate_content(
                (
                    "أنت مساعد مصري ودود. أعد صياغة النص التالي بطريقة ودودة ومصرية حلوة، "
                    "استعمل تعبيرات مثل 'يا باشا'، 'يا أسطى'، 'ربنا يسهّلك'، 'رحلة آمنة'، 'إن شاء الله توصل بالسلامة'. "
                    "احتفظ بجميع المعلومات الفعلية (الأسعار، الأوقات، المسارات) كما هي تمامًا ولا تغير أي رقم أو اسم مكان. "
                    "فقط حسّن الأسلوب واجعل النبرة أكثر ودًا. لا تضف كلمات زيادة ولا تحذف معلومات.\n\n"
                    f"النص:\n{text}"
                ),
                request_options={"retry": None, "timeout": 25}
            )
            polished = getattr(resp, "text", "").strip()
            if polished:
                return polished
        except Exception:
            pass
        
        # Fallback to deprecated google.generativeai
        try:
            import google.generativeai as genai_legacy
            genai_legacy.configure(api_key=api_key)
            model = genai_legacy.GenerativeModel("gemini-1.5-flash")
            resp = model.generate_content(
                (
                    "أنت مساعد مصري ودود. أعد صياغة النص التالي بطريقة ودودة ومصرية حلوة، "
                    "استعمل تعبيرات مثل 'يا باشا'، 'يا أسطى'، 'ربنا يسهّلك'، 'رحلة آمنة'، 'إن شاء الله توصل بالسلامة'. "
                    "احتفظ بجميع المعلومات الفعلية (الأسعار، الأوقات، المسارات) كما هي تمامًا ولا تغير أي رقم أو اسم مكان. "
                    "فقط حسّن الأسلوب واجعل النبرة أكثر ودًا. لا تضف كلمات زيادة ولا تحذف معلومات.\n\n"
                    f"النص:\n{text}"
                ),
                request_options={"timeout": 25}
            )
            polished = resp.text.strip()
            if polished:
                return polished
        except Exception:
            pass
        
    except Exception:
        pass
    
    return text


def _format_response(resp: Dict[str, Any], origin: str, dest: str) -> str:
    if not resp or resp.get("num_journeys", 0) == 0:
        return "لم يتم العثور على رحلات مناسبة بالقرب من نقطتي البداية أو النهاية ضمن مسافة المشي المحددة."

    journeys = resp.get("journeys", [])

    # Sort best: money, then walk
    journeys_sorted = sorted(journeys, key=lambda j: (j.get("costs", {}).get("money", 0), j.get("costs", {}).get("walk", 0)))
    journeys_top = journeys_sorted[:5]

    out = [f"من {origin} إلى {dest}\n"]
    for i, j in enumerate(journeys_top, 1):
        path = j.get("path", [])
        costs = j.get("costs", {})
        readable = [decode_trip(t) for t in path]
        path_text = " → ".join(readable) if readable else "(مسار غير معروف)"
        money = costs.get("money", 0)
        walk_m = int(costs.get("walk", 0))
        time_min = int(costs.get("transport_time", 0))
        out.append(
            (
                f"\n🔹 الرحلة {i}:\n"
                f"🛣 المسار: {path_text}\n"
                f"💰 السعر التقريبي: {money} جنيه\n"
                f"🚶‍♂️ إجمالي المشي: {walk_m} متر\n"
                f"⏱ زمن التنقل: ~{time_min} دقيقة\n"
            )
        )
    out.append("\nنصيحة: اختار اللي يناسبك بين الأقل سعرًا والأقل مشيًا أو الأقل تبديلات.")
    return "".join(out)


def node_format(state: AgentState) -> AgentState:
    resp = state.get("route_response")
    if not resp:
        return state
    origin = state.get("origin") or ""
    dest = state.get("destination") or ""
    raw_formatted = _format_response(resp, origin, dest)
    # Polish with friendly Egyptian tone
    polished = _gemini_polish(raw_formatted)
    state["formatted"] = polished
    return state


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("parse", node_parse)
    graph.add_node("geocode", node_geocode)
    graph.add_node("route", node_route)
    graph.add_node("format", node_format)

    graph.set_entry_point("parse")
    graph.add_edge("parse", "geocode")
    graph.add_edge("geocode", "route")
    graph.add_edge("route", "format")
    graph.add_edge("format", END)

    return graph.compile()


def run_once(query: str) -> str:
    app = build_graph()
    # Initial state
    state: AgentState = {
        "query": query,
        "origin": None,
        "destination": None,
        "origin_geo": None,
        "destination_geo": None,
        "route_response": None,
        "formatted": None,
    }
    final = app.invoke(state)
    formatted = final.get("formatted")
    if not formatted:
        # Fallback message
        return "تعذّر معالجة الطلب بالكامل. تأكّد من كتابة الصيغة بوضوح أو جرّب استعلامًا آخر."
    return formatted
