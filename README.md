# House Price Prediction

A full-stack machine learning web application that predicts house prices based on **area (sq ft)** and **number of rooms**.

| Layer    | Folder      | Technology                                        |
|----------|-------------|---------------------------------------------------|
| Data     | `data/`     | CSV dataset (47 samples)                          |
| ML Model | `model/`    | scikit-learn — Linear Regression + StandardScaler |
| Backend  | `backend/`  | Flask REST API (port 5000)                        |
| Frontend | `frontend/` | Streamlit + Plotly (port 8501)                    |
| Reports  | `reports/`  | Static HTML project report                        |

---

## Project Structure

```
house-price-prediction/
├── data/
│   └── house_price.csv        ← raw dataset (area, rooms, price)
├── model/
│   ├── model.py               ← train & save the ML model
│   └── model.pkl              ← serialised model (auto-generated)
├── backend/
│   └── app.py                 ← Flask REST API on port 5000
├── frontend/
│   └── frontend.py            ← Streamlit UI on port 8501
├── reports/
│   └── report.html            ← open in browser for project report
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python model/model.py
```
Generates `model/model.pkl`.

### 3. Start the Flask API  *(Terminal 1)*
```bash
python backend/app.py
```
API runs at → http://localhost:5000

### 4. Start the Streamlit frontend  *(Terminal 2)*
```bash
streamlit run frontend/frontend.py
```
UI opens at → http://localhost:8501

---

## API Endpoints

| Method | Endpoint        | Description                        |
|--------|-----------------|------------------------------------|
| GET    | `/health`       | API liveness check                 |
| POST   | `/predict`      | Predict price for area + rooms     |
| GET    | `/model-info`   | Model coefficients and metrics     |
| GET    | `/data-summary` | Dataset statistics and all records |

### Example: POST /predict
```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"area": 2000, "rooms": 3}'
```
Response:
```json
{
  "area": 2000.0,
  "rooms": 3,
  "predicted_price": 330662.99,
  "currency": "USD"
}
```

---

## Model Performance

| Metric | Value   |
|--------|---------|
| MAE    | $72,335 |
| RMSE   | $92,792 |
| R²     | 0.5149  |

Trained on 80% of data (37 samples), evaluated on 20% (10 samples).

---

## Report

Open `reports/report.html` in any browser for a standalone project one-pager.
