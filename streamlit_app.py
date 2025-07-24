import streamlit as st
import pandas as pd
import numpy as np
from utils.data_loader import load_and_prepare_data
from utils.features import compute_technical_indicators
from models.lstm_model import train_predict_lstm
from models.arima_model import train_predict_arima
from models.xgboost_model import train_predict_xgboost
from utils.shap_explainer import explain_with_shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Multimodal Stock Price Prediction", layout="wide")

st.title("📈 Multimodal Stock Price Prediction (MSFT / TSLA)")
st.markdown("This app uses an ensemble of BiLSTM, ARIMA, and XGBoost to forecast short-term stock prices using both market and Google Trends data.")

# Select stock
stock = st.selectbox("Select Stock", ["MSFT", "TSLA"])

# Select inputs
st.sidebar.title("Input Features")
use_ohlcv = st.sidebar.checkbox("Use OHLCV data", value=True)
use_trends = st.sidebar.checkbox("Use Google Trends", value=True)
use_derived = st.sidebar.checkbox("Use Derived Technical Indicators (RSI, EMA, etc.)", value=True)

# Set ensemble weightage
st.sidebar.title("Model Weights (Total = 1.0)")
w_lstm = st.sidebar.slider("BiLSTM Weight", 0.0, 1.0, 0.33)
w_arima = st.sidebar.slider("ARIMA Weight", 0.0, 1.0, 0.33)
w_xgb = st.sidebar.slider("XGBoost Weight", 0.0, 1.0, 0.34)

if not np.isclose(w_lstm + w_arima + w_xgb, 1.0):
    st.error("The weights must sum to 1.0")
    st.stop()

# Load and prepare data
df = load_and_prepare_data(stock)
df = compute_technical_indicators(df)

# Feature selection
input_features = []
if use_ohlcv:
    input_features += ["Open", "High", "Low", "Close", "Volume"]
if use_trends:
    input_features += ["Trend"]
if use_derived:
    input_features += ["RSI", "EMA", "MA"]

# Train-test split
train_df = df[:-7]
test_df = df[-7:]

# Run models
st.markdown("## 🔧 Model Predictions")

preds = {}

if use_ohlcv or use_trends or use_derived:
    X_train = train_df[input_features]
    X_test = test_df[input_features]
    y_train = train_df["Close"]
    y_test = test_df["Close"]

    preds["BiLSTM"] = train_predict_lstm(X_train, y_train, X_test)
    preds["ARIMA"] = train_predict_arima(y_train, steps=7)
    preds["XGBoost"] = train_predict_xgboost(X_train, y_train, X_test)

    # Ensemble prediction
    ensemble_pred = (w_lstm * preds["BiLSTM"] +
                     w_arima * preds["ARIMA"] +
                     w_xgb * preds["XGBoost"])

    # Plotting
    st.subheader("📊 Actual vs Predicted")
    fig, ax = plt.subplots()
    ax.plot(test_df.index, y_test, label="Actual")
    ax.plot(test_df.index, ensemble_pred, label="Ensemble Prediction")
    ax.legend()
    st.pyplot(fig)

    # Show each model's prediction
    st.markdown("### 📌 Individual Model Outputs")
    for model_name, pred in preds.items():
        st.write(f"{model_name} prediction:", np.round(pred, 2).tolist())

    # SHAP Explanation
    st.subheader("🔍 Model Explanation (SHAP)")
    shap_text = explain_with_shap(X_train, X_test, model=preds["XGBoost"])
    st.text(shap_text)

else:
    st.warning("Please select at least one data source.")
