import os
import sys
import streamlit as st
import datetime
import pandas as pd

# Add 'utils' folder to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
from stock_forecast_pipeline import StockForecastPipeline

st.set_page_config(page_title="📈 Multimodal Stock Forecast", layout="wide")
st.title("📊 Multimodal Ensemble Stock Forecasting Dashboard")

# Sidebar Inputs
st.sidebar.header("⚙️ Forecast Configuration")
ticker = st.sidebar.text_input("NYSE Stock Ticker (1Y history)", value="AAPL").upper()
arima_weight_pct = st.sidebar.slider("ARIMA Weight (%)", 0, 100, 50)
arima_weight = arima_weight_pct / 100
xgb_weight = 1 - arima_weight
forecast_days = 5

# Dates
end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=1 * 365)

# Session State
if "feature_sets" not in st.session_state:
    st.session_state.feature_sets = {}
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "df" not in st.session_state:
    st.session_state.df = None

# Step 1: Get Features
if st.sidebar.button("🔍 Get Features"):
    try:
        with st.spinner("Loading data and extracting features..."):
            pipeline = StockForecastPipeline(ticker, start_date, end_date, arima_weight)
            pipeline.download_ohlcv_data()
            ohlcv_df = pd.read_csv(pipeline.ohlcv_path, index_col=0, parse_dates=True)
            pipeline.generate_google_trend_data(ohlcv_df)
            pipeline.derive_features()
            df = pipeline.load_data()

            all_features = [col for col in df.columns if col.lower() != "open"]

            ohlcv_features = ["high", "low", "close", "volume"]
            ohlcv_derived = [f for f in all_features if f in ["return_1d", "ema_20", "volatility_10d"]]
            gt_features = [f for f in all_features if f == f"{ticker}_trend"]
            gt_derived = [f for f in all_features if f in ["trend_7d_ma", "trend_rolling_max_50d", "trend_return"]]

            st.session_state.feature_sets = {
                "ohlcv": ohlcv_features,
                "ohlcv_derived": ohlcv_derived,
                "gt": gt_features,
                "gt_derived": gt_derived
            }
            st.session_state.pipeline = pipeline
            st.session_state.df = df
        st.success("Feature sets loaded. Please select features below.")
    except Exception as e:
        st.error(f"Error: {e}")

# Step 2: Show Feature Selection if features loaded
if st.session_state.feature_sets:
    st.markdown("## 🧩 Select Features for Prediction")

    selected_ohlcv = st.multiselect("OHLCV Features", st.session_state.feature_sets["ohlcv"], default=["open", "high"])
    selected_ohlcv_derived = st.multiselect("OHLCV Derived Features", st.session_state.feature_sets["ohlcv_derived"], default=st.session_state.feature_sets["ohlcv_derived"])
    selected_gt = st.multiselect("Google Trend Features", st.session_state.feature_sets["gt"], default=st.session_state.feature_sets["gt"])
    selected_gt_derived = st.multiselect("Google Trend Derived Features", st.session_state.feature_sets["gt_derived"], default=st.session_state.feature_sets["gt_derived"])

    selected_features = selected_ohlcv + selected_ohlcv_derived + selected_gt + selected_gt_derived

    # Step 3: Prediction
    if st.button("✅ Get Prediction"):
        if not selected_features:
            st.warning("Please select at least one feature to proceed.")
        else:
            try:
                with st.spinner("Running ensemble prediction pipeline..."):
                    data = st.session_state.pipeline.train_test_split_rolling(
                        st.session_state.df,
                        selected_features,
                        n_forecasts=forecast_days
                    )
                    result_df = st.session_state.pipeline.predict(data)
                    rmse_arima, rmse_xgb, rmse_ensemble = st.session_state.pipeline.evaluate(result_df)

                    st.subheader(f"📈 Forecast Results (Last {forecast_days} Trading Days)")
                    st.dataframe(result_df.style.format("{:.2f}"))

                    st.markdown("### 📉 Prediction Plot")
                    st.session_state.pipeline.plot(result_df)

                    st.markdown("### 🧠 Feature Importance")
                    fi_df = st.session_state.pipeline.feature_importance()
                    st.dataframe(fi_df.style.format("{:.2f}"))
                    st.session_state.pipeline.plot_feature_importance(fi_df)

            except Exception as e:
                st.error(f"Prediction Error: {e}")
