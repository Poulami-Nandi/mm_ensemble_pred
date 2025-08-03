import os
import sys
import streamlit as st
import datetime
import pandas as pd
import matplotlib.pyplot as plt

# Ensure utils path is included for module import
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
from stock_forecast_pipeline import StockForecastPipeline

st.set_page_config(page_title="Multimodal Ensemble Stock Forecast", layout="wide")

st.markdown("## 📈 Multimodal Ensemble Stock Forecasting")
st.markdown("Use ARIMA + XGBoost on OHLCV and Google Trends features to predict the stock's **Open** price.")

with st.sidebar:
    st.header("🧮 Forecast Configuration")
    ticker = st.text_input("NYSE Stock Ticker (1Y history)", "AAPL")
    arima_weight = st.slider("ARIMA Weight (%)", min_value=0, max_value=100, value=50)
    get_feat = st.button("🔍 Get Features")

if get_feat:
    st.session_state.pipeline = StockForecastPipeline(
        ticker=ticker,
        start_date=(datetime.datetime.today() - datetime.timedelta(days=365)).strftime("%Y-%m-%d"),
        end_date=datetime.datetime.today().strftime("%Y-%m-%d"),
        arima_weight=arima_weight / 100.0
    )

    st.session_state.pipeline.download_ohlcv_data()
    ohlcv_df = pd.read_csv(st.session_state.pipeline.ohlcv_path, index_col=0, parse_dates=True)
    st.session_state.pipeline.generate_google_trend_data(ohlcv_df)
    st.session_state.pipeline.derive_features()
    df = st.session_state.pipeline.load_data()
    st.session_state.df = df

    # Get all features by category
    all_features = df.columns.tolist()
    st.session_state.ohlcv = [col for col in ['open', 'high', 'low', 'close', 'volume'] if col in all_features and col != 'open']
    st.session_state.ohlcv_der = [col for col in all_features if col not in st.session_state.ohlcv and '_trend' not in col and 'trend_' not in col]
    st.session_state.gt = [col for col in all_features if col.endswith('_trend')]
    st.session_state.gt_derived = [col for col in all_features if col.startswith('trend_') and not col.endswith('_trend')]

    st.success("✅ Feature sets loaded. Please select features below.")

# Render feature selectors if data is loaded
if 'df' in st.session_state:
    st.markdown("### 🧠 Select Features for Prediction")

    # Validate default values
    def safe_defaults(candidates, options):
        return [c for c in candidates if c in options]

    col1, col2 = st.columns(2)
    with col1:
        selected_ohlcv = st.multiselect("OHLCV Features",
                                        options=st.session_state.ohlcv,
                                        default=safe_defaults(["high", "low"], st.session_state.ohlcv),
                                        key="ohlcv")
        selected_ohlcv_derived = st.multiselect("OHLCV Derived Features",
                                                options=st.session_state.ohlcv_der,
                                                default=safe_defaults(["return_1d", "ema_20", "volatility_10d"], st.session_state.ohlcv_der),
                                                key="ohlcv_der")
    with col2:
        selected_gt = st.multiselect("Google Trend Features",
                                     options=st.session_state.gt,
                                     default=safe_defaults([f"{ticker.upper()}_trend"], st.session_state.gt),
                                     key="gt")
        selected_gt_derived = st.multiselect("Google Trend Derived Features",
                                             options=st.session_state.gt_derived,
                                             default=safe_defaults(["trend_7d_ma", "trend_rolling_max_50d", "trend_return"], st.session_state.gt_derived),
                                             key="gt_der")

    all_selected_features = selected_ohlcv + selected_ohlcv_derived + selected_gt + selected_gt_derived

    if st.button("✅ Get Prediction", use_container_width=True):
        with st.spinner("Running forecasts and training models..."):
            pipeline = st.session_state.pipeline
            df = st.session_state.df

            data = pipeline.train_test_split_rolling(df, selected_features=all_selected_features)
            result_df = pipeline.predict(data)
            rmse_arima, rmse_xgb, rmse_ensemble = pipeline.evaluate(result_df)
            importance_df = pipeline.compute_feature_importance()

        st.markdown("### 📊 Forecast Results (Last 5 Trading Days)")
        st.dataframe(result_df.round(2))

        st.markdown("### 📉 Prediction Plot")
        fig, ax = plt.subplots()
        result_df.set_index("date")[["actual", "arima_pred", "xgb_pred", "ensemble_pred"]].plot(ax=ax)
        ax.set_ylabel("Open Price")
        ax.set_title(f"{ticker.upper()} Open Price Forecast")
        st.pyplot(fig)

        st.markdown("### 🧠 Feature Importance")
        st.dataframe(importance_df)

        st.markdown("### 📌 RMSE Scores")
        st.metric("ARIMA RMSE", f"{rmse_arima:.4f}")
        st.metric("XGBoost RMSE", f"{rmse_xgb:.4f}")
        st.metric("Ensemble RMSE", f"{rmse_ensemble:.4f}")
