# services/pricing.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.trip_price_class import TripPricePredictor
from models.trip_price_class import load_model
from services.distance import get_distance  

# from models import trip_price_model 
# from models.trip_price_model import TripPricePredictor
joblib_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "trip_price_model.joblib")

PRICE_MODEL = None

def get_cost(trip_id, start_stop, end_stop):
    global PRICE_MODEL
    if PRICE_MODEL is None:
        try:
            PRICE_MODEL = load_model(joblib_path)
        except Exception:
            # Fallback: simple proportional estimation if model missing
            PRICE_MODEL = TripPricePredictor(model=None)

    distance = get_distance(trip_id, start_stop, end_stop)
    if distance is None:
        return 0.0

    if getattr(PRICE_MODEL, "model", None) is None:
        # naive fallback: 3 EGP base + 0.5 per km, bus-style rounding
        base = 3.0
        per_km = 0.5
        raw = base + per_km * float(distance)
        return float(TripPricePredictor(model=None)._round_bus_style(raw))

    price = PRICE_MODEL.predict([distance])[0]
    return float(price)
