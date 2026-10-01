"""
model/model.py — Train and save the house price prediction model.
Run this script once before starting the Flask server:
    python model/model.py
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Resolve absolute paths relative to project root ───────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH    = os.path.join(PROJECT_ROOT, "data",  "house_price.csv")
MODEL_PATH   = os.path.join(PROJECT_ROOT, "model", "model.pkl")

# ── 1. Load dataset ────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(df.describe())

# ── 2. Features & target ───────────────────────────────────────────────────────
X = df[["area", "rooms"]].values
y = df["price"].values

# ── 3. Train / test split ──────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 4. Feature scaling ─────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── 5. Train model ─────────────────────────────────────────────────────────────
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# ── 6. Evaluate ────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test_scaled)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print("\n-- Model Evaluation --")
print(f"  MAE  : ${mae:,.2f}")
print(f"  RMSE : ${rmse:,.2f}")
print(f"  R2   : {r2:.4f}")
print(f"  Coefficients: area={model.coef_[0]:.2f}, rooms={model.coef_[1]:.2f}")
print(f"  Intercept   : {model.intercept_:.2f}")

# ── 7. Save model + scaler ─────────────────────────────────────────────────────
with open(MODEL_PATH, "wb") as f:
    pickle.dump({"model": model, "scaler": scaler}, f)

print(f"\nModel saved to {MODEL_PATH}")

# ── 8. Helpers re-exported for use by backend/app.py ──────────────────────────
def load_artifacts():
    """Load and return (model, scaler) from model/model.pkl."""
    with open(MODEL_PATH, "rb") as f:
        artifacts = pickle.load(f)
    return artifacts["model"], artifacts["scaler"]


def predict_price(area: float, rooms: int) -> float:
    """Return predicted price for given area (sq ft) and rooms."""
    mdl, scl = load_artifacts()
    X_new = scl.transform([[area, rooms]])
    return float(mdl.predict(X_new)[0])


if __name__ == "__main__":
    sample = predict_price(2000, 3)
    print(f"\nSample prediction (area=2000, rooms=3): ${sample:,.2f}")
