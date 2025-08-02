import sys
import os

# Add 'utils' directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from stock_forecast_pipeline import StockForecastPipeline
import streamlit as st
import datetime
import pandas as pd
import matplotlib.pyplot as plt

# Streamlit configuration
st.set_page_config(page_title="📊 Stock Forecasting", layout="wide")
st.title("📈 Stock Price Forecasting using ARIMA, XGBoost & Ensemble")

# Sidebar inputs
st.sidebar.header("Configure Forecast")
ticker = st.sidebar.text_input("Enter NYSE Stock Ticker", "AAPL").strip().upper()
arima_weight_pct = st.sidebar.slider("ARIMA Model Weight (%)", 0, 100, 50)

feature_list = [
    "return_1d", "ema_20", "volatility_10d",
    "trend_7d_ma", "trend_rolling_max_50d", "trend_return",
    "high", "low", "close", "volume"
]
selected_features = st.sidebar.multiselect(
    "Select Features to Include",
    options=feature_list,
    default=[
        "return_1d", "ema_20", "volatility_10d",
        "trend_7d_ma", "trend_rolling_max_50d", "trend_return"
    ]
)

if st.sidebar.button("🔍 Run Forecast"):
    try:
        arima_weight = arima_weight_pct / 100
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=5 * 365)
        forecast_days = 5

        with st.spinner("Processing data and training models..."):
            pipeline = StockForecastPipeline(ticker, start_date, end_date, arima_weight=arima_weight)
            pipeline.download_ohlcv_data()
            df_ohlcv = pd.read_csv(pipeline.ohlcv_path, index_col=0, parse_dates=True)
            pipeline.generate_google_trend_data(df_ohlcv)
            pipeline.derive_features()
            df = pipeline.load_data()

            data = pipeline.train_test_split_rolling(df, selected_features, n_forecasts=forecast_days)
            result_df = pipeline.predict(data)
            rmse_arima, rmse_xgb, rmse_ensemble = pipeline.evaluate(result_df)

        st.subheader(f"📊 Forecast for Last {forecast_days} Trading Days")
        st.dataframe(result_df.style.format("{:.2f}"))

        # Plot forecast
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(result_df.index, result_df["actual"], label="Actual", color="black", linewidth=2)
        ax1.plot(result_df.index, result_df["arima_pred"], label="ARIMA", linestyle="--")
        ax1.plot(result_df.index, result_df["xgb_pred"], label="XGBoost", linestyle="-.", color="orangered")
        ax1.plot(result_df.index, result_df["ensemble_pred"], label="Ensemble", color="green")
        ax1.set_title(f"{ticker} Price Forecast (Last {forecast_days} Days)")
        ax1.set_xticks(result_df.index)
        ax1.set_xticklabels([d.strftime("%Y-%m-%d") for d in result_df.index], rotation=45)
        ax1.legend()
        ax1.grid(True)
        st.pyplot(fig1)

        # Feature importance
        st.subheader("📌 Feature Importance")
        feat_df = pipeline.feature_importance()
        st.dataframe(feat_df)

        fig2, ax2 = plt.subplots(figsize=(10, 5))
        feat_df.plot(kind="bar", ax=ax2)
        ax2.set_ylabel("Importance (%)")
        ax2.set_xticklabels(feat_df.index, rotation=45)
        ax2.set_title("Feature Contribution by Model")
        ax2.grid(True)
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"Something went wrong: {e}")
        st.exception(e)
