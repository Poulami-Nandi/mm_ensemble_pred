import streamlit as st
import os
import sys
import datetime
import pandas as pd
import matplotlib.pyplot as plt

# Add 'utils' directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from stock_forecast_pipeline import StockForecastPipeline

st.set_page_config(page_title="📈 Multimodal Ensemble Stock Forecast", layout="wide")
st.title("📈 Multimodal Ensemble Stock Forecast")

# Sidebar: Forecast Configuration
st.sidebar.header("📊 Forecast Configuration")
ticker = st.sidebar.text_input("NYSE Stock Ticker (1Y history)", value="AAPL")
arima_weight = st.sidebar.slider("ARIMA Weight (%)", 0, 100, 50)

# Button: Get Features
if st.sidebar.button("📂 Get Features"):
    with st.spinner("Loading feature sets..."):
        start_date = (datetime.datetime.today() - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
        end_date = datetime.datetime.today().strftime("%Y-%m-%d")

        pipeline = StockForecastPipeline(ticker, start_date, end_date, arima_weight)
        pipeline.download_ohlcv_data()
        ohlcv_df = pd.read_csv(pipeline.ohlcv_path, index_col=0, parse_dates=True)
        pipeline.generate_google_trend_data(ohlcv_df)
        pipeline.derive_features()
        df = pipeline.load_data()

        st.session_state.pipeline = pipeline
        st.session_state.df = df
        st.session_state.available_features = df.columns.tolist()

        # Divide into 4 groups
        st.session_state.ohlcv_features = ['open', 'high', 'low', 'close', 'volume']
        st.session_state.ohlcv_derived = [f for f in df.columns if f not in st.session_state.ohlcv_features and "_trend" not in f]
        st.session_state.gt_features = [f for f in df.columns if f.lower().endswith("_trend") and "derived" not in f]
        st.session_state.gt_derived = [f for f in df.columns if "_trend" in f and f not in st.session_state.gt_features]

        st.success("Feature sets loaded. Please select features below.")

# Feature Selection UI
if "available_features" in st.session_state:
    st.subheader("🧠 Select Features for Prediction")

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    with col1:
        selected_ohlcv = st.multiselect("OHLCV Features", options=st.session_state.ohlcv_features,
                                        default=["low", "high"], label_visibility="visible", key="ohlcv")

    with col2:
        selected_ohlcv_derived = st.multiselect("OHLCV Derived Features", options=st.session_state.ohlcv_derived,
                                                default=["return_1d", "ema_20", "volatility_10d"],
                                                label_visibility="visible", key="ohlcv_der")

    with col3:
        selected_gt = st.multiselect("Google Trend Features", options=st.session_state.gt_features,
                                     default=[f"{ticker.upper()}_trend"], label_visibility="visible", key="gt")

    with col4:
        selected_gt_derived = st.multiselect("Google Trend Derived Features", options=st.session_state.gt_derived,
                                             default=["trend_7d_ma", "trend_rolling_max_50d", "trend_return"],
                                             label_visibility="visible", key="gt_der")

    selected_features = selected_ohlcv + selected_ohlcv_derived + selected_gt + selected_gt_derived

    if st.button("✅ Get Prediction", use_container_width=True):
        if not selected_features:
            st.warning("Please select at least one feature.")
        else:
            with st.spinner("Running forecasts and training models..."):
                pipeline = st.session_state.pipeline
                df = st.session_state.df

                data = pipeline.train_test_split_rolling(df, selected_features)
                result_df = pipeline.predict(data)
                rmse_arima, rmse_xgb, rmse_ensemble = pipeline.evaluate(result_df)
                fi_df = pipeline.get_feature_importance(result_df, selected_features)

            # Forecast Table
            st.subheader(f"📊 Forecast Results (Last {len(result_df)} Trading Days)")
            st.dataframe(result_df.style.format("{:.2f}"))

            # RMSE
            st.markdown("### 📉 RMSE Scores")
            st.write(f"**ARIMA RMSE:** {rmse_arima:.4f}")
            st.write(f"**XGBoost RMSE:** {rmse_xgb:.4f}")
            st.write(f"**Ensemble RMSE:** {rmse_ensemble:.4f}")

            # Plot
            st.markdown("### 📈 Prediction Plot")
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(result_df["date"], result_df["actual"], label="Actual", color="black")
            ax.plot(result_df["date"], result_df["arima_pred"], label="ARIMA", linestyle="--")
            ax.plot(result_df["date"], result_df["xgb_pred"], label="XGBoost", linestyle="dashdot")
            ax.plot(result_df["date"], result_df["ensemble_pred"], label="Ensemble", color="green")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.set_title(f"{ticker.upper()} Price Forecast (Last {len(result_df)} Days)")
            ax.legend()
            ax.tick_params(axis="x", rotation=45)
            st.pyplot(fig)

            # Feature Importance
            st.markdown("### 🧠 Feature Importance")
            st.dataframe(fi_df.style.format("{:.2f}"))
