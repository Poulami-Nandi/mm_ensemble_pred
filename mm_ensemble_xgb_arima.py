import sys
import os

import streamlit as st
import datetime
import pandas as pd
import matplotlib.pyplot as plt

# Add 'utils' folder to path for import
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from stock_forecast_pipeline import StockForecastPipeline

st.set_page_config(page_title="📈 Multimodal Stock Forecast", layout="wide")

st.title("📊 Multimodal Ensemble Stock Forecasting Dashboard")

# Sidebar Inputs
st.sidebar.header("⚙️ Forecast Configuration")
ticker = st.sidebar.text_input("NYSE Stock Ticker (5Y history)", value="AAPL").upper()
arima_weight_pct = st.sidebar.slider("ARIMA Weight (%)", 0, 100, 1)
arima_weight = arima_weight_pct / 100
xgb_weight = 1 - arima_weight
forecast_days = 5

# Dates
end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=5 * 365)

if st.sidebar.button("🔍 Run Forecast"):
    try:
        with st.spinner("Running multimodal pipeline..."):
            pipeline = StockForecastPipeline(ticker, start_date, end_date, arima_weight)
            pipeline.download_ohlcv_data()
            ohlcv_df = pd.read_csv(pipeline.ohlcv_path, index_col=0, parse_dates=True)
            pipeline.generate_google_trend_data(ohlcv_df)
            pipeline.derive_features()
            df = pipeline.load_data()

            all_features = df.columns.tolist()

            # Separate features by type
            ohlcv_features = ["open", "high", "low", "close", "volume"]
            ohlcv_derived_features = [f for f in all_features if f in ["return_1d", "ema_20", "volatility_10d"]]
            trend_features = [f for f in all_features if f == f"{ticker}_trend"]
            trend_derived_features = [f for f in all_features if f in ["trend_7d_ma", "trend_rolling_max_50d", "trend_return"]]

            st.sidebar.markdown("### 📌 Feature Selection")

            selected_ohlcv = st.sidebar.multiselect("OHLCV Data", ohlcv_features, default=ohlcv_features[:3])
            selected_ohlcv_der = st.sidebar.multiselect("OHLCV Derived", ohlcv_derived_features, default=ohlcv_derived_features)
            selected_gt = st.sidebar.multiselect("Google Trend", trend_features, default=trend_features)
            selected_gt_der = st.sidebar.multiselect("Google Trend Derived", trend_derived_features, default=trend_derived_features)

            selected_features = selected_ohlcv + selected_ohlcv_der + selected_gt + selected_gt_der

            if not selected_features:
                st.warning("Please select at least one feature to continue.")
            else:
                rolling_data = pipeline.train_test_split_rolling(df, selected_features, n_forecasts=forecast_days)
                result_df = pipeline.predict(rolling_data)
                rmse_arima, rmse_xgb, rmse_ensemble = pipeline.evaluate(result_df)

                st.subheader(f"📈 Forecast Results for Last {forecast_days} Trading Days")
                st.dataframe(result_df.style.format("{:.2f}"))

                st.markdown("### 📉 Prediction Plot")
                pipeline.plot(result_df)

                st.markdown("### 🧠 Feature Importance")
                fi_df = pipeline.feature_importance()
                st.dataframe(fi_df.style.format("{:.2f}"))
                pipeline.plot_feature_importance(fi_df)

    except Exception as e:
        st.error(f"🚨 Error: {e}")
