import sys
import os

import streamlit as st
import datetime
import pandas as pd
import matplotlib.pyplot as plt

# Add 'utils' directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
from stock_forecast_pipeline import StockForecastPipeline

st.set_page_config(layout="wide")
st.title("📈 Multimodal Stock Price Forecast using ARIMA + XGBoost Ensemble")

# --- User Inputs ---
ticker = st.text_input("Enter NYSE Stock Ticker:", value="AAPL")
arima_weight_pct = st.slider("ARIMA Weight (%)", min_value=0, max_value=100, value=50)

arima_weight = arima_weight_pct / 100
xgb_weight = 1 - arima_weight

st.markdown(f"**Using ARIMA weight:** {arima_weight:.2f}, **XGBoost weight:** {xgb_weight:.2f}")

end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=1 * 365)

# --- Feature Selection UI ---
st.header("🧠 Feature Selection by Data Type")

with st.expander("📊 OHLCV Raw Data Features"):
    ohlcv_raw_feats = st.multiselect("Select OHLCV Raw Features:", ["open", "high", "low", "close", "volume"], default=["high", "low"])

with st.expander("📈 OHLCV Derived Features"):
    ohlcv_derived_feats = st.multiselect("Select OHLCV Derived Features:", ["return_1d", "ema_20", "volatility_10d"], default=["return_1d", "ema_20"])

with st.expander("🔍 Google Trend Raw Data"):
    trend_raw_feats = st.multiselect("Select Google Trend Raw Feature:", [f"{ticker.upper()}_trend"], default=[])

with st.expander("🧪 Google Trend Derived Features"):
    trend_derived_feats = st.multiselect("Select Google Trend Derived Features:", ["trend_7d_ma", "trend_rolling_max_50d", "trend_return"], default=["trend_7d_ma"])

selected_features = ohlcv_raw_feats + ohlcv_derived_feats + trend_raw_feats + trend_derived_feats

if not selected_features:
    st.warning("Please select at least one feature to proceed.")
    st.stop()

# --- Run Forecast Pipeline ---

if st.button("Run Forecast", use_container_width=True):
    with st.spinner("Running forecasts and training models..."):
        pipeline = StockForecastPipeline(ticker, start_date, end_date, arima_weight=arima_weight)
        pipeline.download_ohlcv_data()
        ohlcv_df = pd.read_csv(pipeline.ohlcv_path, index_col=0, parse_dates=True)
        pipeline.generate_google_trend_data(ohlcv_df)
        pipeline.derive_features()
        df = pipeline.load_data()

        rolling_data = pipeline.train_test_split_rolling(df, selected_features, n_forecasts=5)
        results = pipeline.predict(rolling_data)
        rmse_arima, rmse_xgb, rmse_ensemble = pipeline.evaluate(results)

        # --- Plot Results ---
        st.subheader("📉 Forecasted vs Actual Price")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(results.index, results["actual"], label="Actual", color="black", linewidth=2)
        ax.plot(results.index, results["arima_pred"], label="ARIMA", linestyle="--")
        ax.plot(results.index, results["xgb_pred"], label="XGBoost", linestyle="-.", color="orangered")
        ax.plot(results.index, results["ensemble_pred"], label="Ensemble", color="green")

        ax.set_title(f"{ticker.upper()} Forecast (Last {len(results)} Days)")
        ax.set_xticks(results.index)
        ax.set_xticklabels([d.strftime("%Y-%m-%d") for d in results.index], rotation=45)
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

        # --- RMSE ---
        st.success(f"RMSE (ARIMA): {rmse_arima:.4f}, XGBoost: {rmse_xgb:.4f}, Ensemble: {rmse_ensemble:.4f}")

        # --- Feature Importance ---
        st.subheader("🔎 Feature Importance Comparison")
        feat_imp_df = pipeline.feature_importance()
        st.dataframe(feat_imp_df.sort_values("Ensemble (%)", ascending=False))

        fig2, ax2 = plt.subplots(figsize=(10, 5))
        feat_imp_df.plot(kind='bar', ax=ax2)
        ax2.set_ylabel("Importance (%)")
        ax2.set_title("Feature Importance by Model")
        ax2.grid(True)
        st.pyplot(fig2)




