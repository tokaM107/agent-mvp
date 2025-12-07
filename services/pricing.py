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

PRICE_MODEL = load_model(joblib_path)

def get_cost(trip_id, start_stop, end_stop):
    distance = get_distance(trip_id, start_stop, end_stop)

    # if distance is None:
    #     return 0  

    price = PRICE_MODEL.predict([distance])[0]
    return float(price)
