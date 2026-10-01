"""
backend/app.py — Flask REST API backend for House Price Prediction.
Run with:  python backend/app.py

Endpoints:
  GET  /health          -> liveness check
  POST /predict         -> { "area": float, "rooms": int } -> { "predicted_price": float }
  GET  /model-info      -> coefficients, intercept, metrics
  GET  /data-summary    -> dataset statistics and all rows
"""

import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app)

# ── Resolve paths from project root ───────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH   = os.path.join(PROJECT_ROOT, "model", "model.pkl")
DATA_PATH    = os.path.join(PROJECT_ROOT, "data",  "house_price.csv")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        artifacts = pickle.load(f)
    return artifacts["model"], artifacts["scaler"]


def _load_data():
    return pd.read_csv(DATA_PATH)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    model_ready = os.path.exists(MODEL_PATH)
    return jsonify({"status": "ok", "model_ready": model_ready}), 200


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    if "area" not in data or "rooms" not in data:
        return jsonify({"error": "Both 'area' and 'rooms' fields are required."}), 400

    try:
        area  = float(data["area"])
        rooms = int(data["rooms"])
    except (ValueError, TypeError):
        return jsonify({"error": "'area' must be a number and 'rooms' must be an integer."}), 400

    if area <= 0 or rooms <= 0:
        return jsonify({"error": "'area' and 'rooms' must be positive values."}), 400

    if not os.path.exists(MODEL_PATH):
        return jsonify({"error": "Model not trained yet. Run: python model/model.py"}), 503

    model, scaler = _load_artifacts()
    X_new = scaler.transform([[area, rooms]])
    price = float(model.predict(X_new)[0])
    price = max(price, 0)

    return jsonify({
        "area": area,
        "rooms": rooms,
        "predicted_price": round(price, 2),
        "currency": "USD"
    }), 200


@app.route("/model-info", methods=["GET"])
def model_info():
    if not os.path.exists(MODEL_PATH):
        return jsonify({"error": "Model not trained yet."}), 503

    model, scaler = _load_artifacts()
    df = _load_data()

    X = df[["area", "rooms"]].values
    y = df["price"].values
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2   = r2_score(y_test, y_pred)

    return jsonify({
        "algorithm": "Linear Regression",
        "features": ["area (sq ft)", "rooms"],
        "coefficients": {
            "area":  round(float(model.coef_[0]), 4),
            "rooms": round(float(model.coef_[1]), 4)
        },
        "intercept": round(float(model.intercept_), 4),
        "metrics": {
            "MAE":  round(mae, 2),
            "RMSE": round(rmse, 2),
            "R2":   round(r2, 4)
        },
        "train_test_split": "80/20",
        "scaling": "StandardScaler"
    }), 200


@app.route("/data-summary", methods=["GET"])
def data_summary():
    df = _load_data()
    desc = df.describe().round(2).to_dict()
    rows = df.to_dict(orient="records")
    return jsonify({
        "total_rows": len(df),
        "columns": list(df.columns),
        "statistics": desc,
        "records": rows
    }), 200


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print("WARNING: model/model.pkl not found. Run 'python model/model.py' first.")
    print("Starting Flask API on http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
