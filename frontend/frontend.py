"""
frontend/frontend.py — Streamlit UI for House Price Prediction.
Run with:  streamlit run frontend/frontend.py
Requires Flask API running on http://localhost:5000
"""

import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = "http://localhost:5000"

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {font-size:2.4rem; font-weight:700; color:#1f2328; margin-bottom:0;}
    .sub-title  {font-size:1rem; color:#57606a; margin-top:0;}
    .price-highlight {
        font-size:2.8rem; font-weight:800; color:#3b82d4;
        background:#f0f6ff; border:2px solid #3b82d4;
        border-radius:12px; padding:20px 32px; text-align:center;
        margin:16px 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def fetch_data_summary():
    try:
        r = requests.get(f"{API_BASE}/data-summary", timeout=5)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)


@st.cache_data(ttl=30)
def fetch_model_info():
    try:
        r = requests.get(f"{API_BASE}/model-info", timeout=5)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)


def call_predict(area: float, rooms: int):
    try:
        r = requests.post(
            f"{API_BASE}/predict",
            json={"area": area, "rooms": rooms},
            timeout=5,
        )
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to API. Make sure `python backend/app.py` is running."
    except Exception as e:
        return None, str(e)


def api_health():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.json().get("status") == "ok"
    except Exception:
        return False


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏠 House Price Predictor")
    st.markdown("---")

    healthy = api_health()
    if healthy:
        st.success("API connected", icon="✅")
    else:
        st.error("API offline — run `python backend/app.py`")

    st.markdown("### Input Features")
    area  = st.slider("Area (sq ft)", min_value=500, max_value=6000, value=2000, step=50)
    rooms = st.slider("Number of Rooms", min_value=1, max_value=8, value=3, step=1)

    st.markdown("---")
    predict_btn = st.button("Predict Price", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown("**API Endpoints**")
    st.code(f"GET  {API_BASE}/health\nPOST {API_BASE}/predict\nGET  {API_BASE}/model-info\nGET  {API_BASE}/data-summary")


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🏠 House Price Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Linear Regression · Flask API + Streamlit UI</p>', unsafe_allow_html=True)
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Data Explorer", "🤖 Model Info"])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("### Selected Inputs")
        c1, c2 = st.columns(2)
        c1.metric("Area (sq ft)", f"{area:,}")
        c2.metric("Rooms", rooms)

        st.markdown("#### Area × Rooms Price Map")
        info, _ = fetch_model_info()
        if info:
            coef_area  = info["coefficients"]["area"]
            coef_rooms = info["coefficients"]["rooms"]
            intercept  = info["intercept"]
            areas_range = np.linspace(500, 5000, 20)
            rooms_range = [1, 2, 3, 4, 5]
            area_mean, area_std   = 2000, 800
            rooms_mean, rooms_std = 3, 0.9
            z = np.array([
                [
                    intercept
                    + coef_area  * ((a - area_mean)  / area_std)
                    + coef_rooms * ((rm - rooms_mean) / rooms_std)
                    for a in areas_range
                ]
                for rm in rooms_range
            ])
            fig_heat = go.Figure(go.Heatmap(
                z=z,
                x=[f"{int(a):,}" for a in areas_range],
                y=[f"{rm}R" for rm in rooms_range],
                colorscale="Blues",
                colorbar=dict(title="Price ($)")
            ))
            fig_heat.update_layout(
                height=220, margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title="Area (sq ft)", yaxis_title="Rooms",
            )
            st.plotly_chart(fig_heat, use_container_width=True)

    with col_result:
        st.markdown("### Prediction Result")
        if predict_btn:
            result, err = call_predict(area, rooms)
            if err:
                st.error(f"Error: {err}")
            elif result:
                price = result["predicted_price"]
                st.markdown(
                    f'<div class="price-highlight">💰 ${price:,.0f}</div>',
                    unsafe_allow_html=True,
                )
                st.success(f"Predicted price for **{area:,} sq ft**, **{rooms} rooms**: **${price:,.0f}**")
                data, _ = fetch_data_summary()
                if data:
                    prices = [r["price"] for r in data["records"]]
                    percentile = round(float(np.mean(np.array(prices) < price)) * 100, 1)
                    st.info(f"Higher than **{percentile}%** of homes in the training dataset.")
        else:
            st.info("Adjust the sliders and click **Predict Price** to get a prediction.")

        data, data_err = fetch_data_summary()
        if data and not data_err:
            df_vis = pd.DataFrame(data["records"])
            fig_scatter = px.scatter(
                df_vis, x="area", y="price", color="rooms",
                size="rooms",
                labels={"area": "Area (sq ft)", "price": "Price ($)", "rooms": "Rooms"},
                title="Training Data — Area vs Price",
                color_continuous_scale="Blues",
            )
            fig_scatter.add_trace(go.Scatter(
                x=[area], y=[0],
                mode="markers",
                marker=dict(symbol="star", size=16, color="#e11d48"),
                name="Your Input",
            ))
            if predict_btn:
                result2, _ = call_predict(area, rooms)
                if result2:
                    fig_scatter.data[-1].y = (result2["predicted_price"],)
            fig_scatter.update_layout(height=320, margin=dict(l=0, r=0, t=36, b=0))
            st.plotly_chart(fig_scatter, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — DATA EXPLORER
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    data, data_err = fetch_data_summary()
    if data_err or not data:
        st.error(f"Could not load data: {data_err}")
    else:
        df = pd.DataFrame(data["records"])

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Samples", data["total_rows"])
        k2.metric("Avg Price",     f"${df['price'].mean():,.0f}")
        k3.metric("Avg Area",      f"{df['area'].mean():,.0f} sq ft")
        k4.metric("Avg Rooms",     f"{df['rooms'].mean():.1f}")

        st.markdown("---")
        col_l, col_r = st.columns(2, gap="large")

        with col_l:
            fig_hist = px.histogram(df, x="price", nbins=15,
                title="Price Distribution",
                labels={"price": "Price ($)"},
                color_discrete_sequence=["#3b82d4"])
            fig_hist.update_layout(height=300, margin=dict(l=0, r=0, t=36, b=0))
            st.plotly_chart(fig_hist, use_container_width=True)

            fig_area = px.scatter(df, x="area", y="price", trendline="ols",
                title="Area vs Price (with trend line)",
                labels={"area": "Area (sq ft)", "price": "Price ($)"},
                color_discrete_sequence=["#3b82d4"])
            fig_area.update_layout(height=300, margin=dict(l=0, r=0, t=36, b=0))
            st.plotly_chart(fig_area, use_container_width=True)

        with col_r:
            fig_bar = px.bar(
                df.groupby("rooms")["price"].mean().reset_index(),
                x="rooms", y="price",
                title="Average Price by Number of Rooms",
                labels={"rooms": "Rooms", "price": "Avg Price ($)"},
                color="rooms", color_continuous_scale="Blues")
            fig_bar.update_layout(height=300, margin=dict(l=0, r=0, t=36, b=0))
            st.plotly_chart(fig_bar, use_container_width=True)

            fig_box = px.box(df, x="rooms", y="price",
                title="Price Spread by Rooms",
                labels={"rooms": "Rooms", "price": "Price ($)"},
                color="rooms",
                color_discrete_sequence=px.colors.sequential.Blues_r)
            fig_box.update_layout(height=300, margin=dict(l=0, r=0, t=36, b=0))
            st.plotly_chart(fig_box, use_container_width=True)

        st.markdown("#### Correlation Matrix")
        corr = df.corr(numeric_only=True).round(3)
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="Blues",
            title="Feature Correlation Heatmap")
        fig_corr.update_layout(height=300, margin=dict(l=0, r=0, t=36, b=0))
        st.plotly_chart(fig_corr, use_container_width=True)

        st.markdown("#### Raw Dataset")
        st.dataframe(df.style.highlight_max(subset=["price"], color="#d1e8ff"), use_container_width=True)

        st.markdown("#### Descriptive Statistics")
        st.dataframe(df.describe().round(2), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL INFO
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    info, info_err = fetch_model_info()
    if info_err or not info:
        st.error(f"Could not load model info: {info_err}")
    else:
        col_m1, col_m2 = st.columns(2, gap="large")

        with col_m1:
            st.markdown("### Model Details")
            st.markdown(f"""
| Property         | Value                      |
|------------------|----------------------------|
| Algorithm        | {info['algorithm']}        |
| Features         | area (sq ft), rooms        |
| Scaling          | {info['scaling']}          |
| Train/Test Split | {info['train_test_split']} |
""")
            st.markdown("### Coefficients")
            coef_df = pd.DataFrame([
                {"Feature": "area (sq ft)", "Coefficient": info["coefficients"]["area"]},
                {"Feature": "rooms",        "Coefficient": info["coefficients"]["rooms"]},
                {"Feature": "intercept",    "Coefficient": info["intercept"]},
            ])
            st.dataframe(coef_df, use_container_width=True)
            fig_coef = px.bar(
                coef_df[coef_df["Feature"] != "intercept"],
                x="Feature", y="Coefficient",
                title="Feature Coefficients (scaled space)",
                color="Coefficient", color_continuous_scale="Blues")
            fig_coef.update_layout(height=280, margin=dict(l=0, r=0, t=36, b=0))
            st.plotly_chart(fig_coef, use_container_width=True)

        with col_m2:
            st.markdown("### Performance Metrics (Test Set)")
            m = info["metrics"]
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("MAE",  f"${m['MAE']:,.0f}",  help="Mean Absolute Error")
            mc2.metric("RMSE", f"${m['RMSE']:,.0f}", help="Root Mean Squared Error")
            mc3.metric("R²",   f"{m['R2']:.4f}",     help="Coefficient of Determination")

            st.info(f"""
- **MAE  ${m['MAE']:,.0f}** — On average, predictions are off by this amount.
- **RMSE ${m['RMSE']:,.0f}** — Penalises large errors more heavily than MAE.
- **R² {m['R2']:.4f}** — Model explains {m['R2']*100:.1f}% of price variance.
""")
            st.markdown("#### Actual vs Predicted (Test Set)")
            data2, _ = fetch_data_summary()
            if data2:
                from sklearn.model_selection import train_test_split as tts
                from sklearn.linear_model import LinearRegression
                from sklearn.preprocessing import StandardScaler
                df2 = pd.DataFrame(data2["records"])
                X = df2[["area", "rooms"]].values
                y = df2["price"].values
                X_tr, X_te, y_tr, y_te = tts(X, y, test_size=0.2, random_state=42)
                sc = StandardScaler(); sc.fit(X_tr)
                lr = LinearRegression(); lr.fit(sc.transform(X_tr), y_tr)
                y_pr = lr.predict(sc.transform(X_te))
                fig_av = go.Figure()
                fig_av.add_trace(go.Scatter(x=list(range(len(y_te))), y=list(y_te),
                    mode="lines+markers", name="Actual", line=dict(color="#1f2328")))
                fig_av.add_trace(go.Scatter(x=list(range(len(y_pr))), y=list(y_pr),
                    mode="lines+markers", name="Predicted", line=dict(color="#3b82d4", dash="dash")))
                fig_av.update_layout(height=280, xaxis_title="Sample Index",
                    yaxis_title="Price ($)", margin=dict(l=0, r=0, t=10, b=0),
                    legend=dict(orientation="h", y=1.1))
                st.plotly_chart(fig_av, use_container_width=True)

        st.markdown("---")
        st.markdown("### Linear Regression Formula")
        st.latex(r"\hat{y} = \beta_0 + \beta_1 \cdot \text{area\_scaled} + \beta_2 \cdot \text{rooms\_scaled}")
        st.markdown(f"""
- **β₀** (intercept) = `{info['intercept']}`
- **β₁** (area) = `{info['coefficients']['area']}`
- **β₂** (rooms) = `{info['coefficients']['rooms']}`
- Features standardised using **StandardScaler** before prediction.
""")
