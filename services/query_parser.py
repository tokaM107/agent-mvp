import re

# Basic synonyms map to improve geocoding
SYNONYMS = {
    "الموقف الجديد": "El-Mawqaf El-Geded",
    "محطة الرمل": "Raml Station",
    "سيدي جابر": "Sidi Gaber",
    "العصافرة": "Asafra",
    "السرايا": "El Saraya",
    "المنشية": "El-Mansheya",
    "الساعة": "Clock Square",
    "محطة مصر": "Train Station El-Shohada Square",
    "العجمي": "El Agamy",
}

def normalize_name(name: str) -> str:
    name = name.strip()
    # Return synonym if exists, otherwise keep original
    return SYNONYMS.get(name, name)

_route_re = re.compile(r"من\s+(.*?)\s+إلى\s+(.*)")

def parse_arabic_route_query(text: str):
    """Extract start and end addresses from Arabic query like 'أريد الذهاب من X إلى Y'."""
    m = _route_re.search(text)
    if not m:
        return None, None
    start = m.group(1).strip()
    end = m.group(2).strip()
    return start, end
