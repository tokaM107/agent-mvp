# models/trip_price_model.py

import numpy as np
from joblib import load
from sklearn.base import BaseEstimator, RegressorMixin
import os

class TripPricePredictor(BaseEstimator, RegressorMixin):
    def __init__(self, model):
        self.model = model

    def _round_bus_style(self, vals):
        scalar = np.isscalar(vals)
        arr = np.array([vals]) if scalar else np.asarray(vals)
        out = []

        for v in arr:
            pounds = int(np.floor(v))
            dec = v - pounds
            if dec < 0.125: r = pounds + 0.0
            elif dec < 0.375: r = pounds + 0.25
            elif dec < 0.75: r = pounds + 0.5
            else: r = pounds + 1.0
            out.append(round(r, 2))

        return out[0] if scalar else np.array(out)

    def predict(self, distance_km):
        X = np.array(distance_km)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_log = np.log1p(X)
        raw_pred = self.model.predict(X_log)
        return self._round_bus_style(raw_pred)

def load_model(joblib_path):
    model = load(joblib_path)
    return TripPricePredictor(model)

# Fix for pickle loading error - Alias for typo in saved model
TripPricePreedictor = TripPricePredictor
