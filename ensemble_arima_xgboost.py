import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from sklearn.exceptions import ConvergenceWarning
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Constants
gt_col_name = "Microsoft stock"

# Streamlit UI
st.title("📈 Ensemble Stock Price Predictor (ARIMA + XGBoost)")
st.write("Forecasts next-day stock **Open** price using a weighted ensemble of ARIMA and XGBoost.")

ohlcv_file = st.file_uploader("Upload OHLCV CSV file", type="csv")
gt_file = st.file_uploader("Upload Google Trends CSV file", type="csv")

all_features = ['open', 'close', 'high', 'low', 'volume', gt_col_name]
selected_features = st.multiselect("Select features to use for prediction", all_features, default=['close', 'volume', gt_col_name])

# Weight sliders
arima_weight = st.slider("ARIMA Weight", 0.0, 1.0, 0.5, 0.05)
xgb_weight = 1 - arima_weight

if ohlcv_file and gt_file and selected_features:
    ohlcv_df = pd.read_csv(ohlcv_file, parse_dates=['date']).set_index('date')
    gt_df = pd.read_csv(gt_file, parse_dates=['Day']).set_index('Day')
    df = pd.merge(ohlcv_df, gt_df, left_index=True, right_index=True, how='inner')
    df = df[[*ohlcv_df.columns, gt_col_name]].dropna()
    
    exog_features = [f for f in selected_features if f != 'open']

    predictions_ensemble, predictions_arima, predictions_xgb, actuals, dates = [], [], [], [], []

    for i in range(5, 0, -1):
        train = df.iloc[:-i].copy()
        test_date = df.iloc[-i:].index[0]

        y_train = pd.to_numeric(train['open'], errors='coerce')
        X_train = train[exog_features].apply(pd.to_numeric, errors='coerce')
        combined = pd.concat([y_train, X_train], axis=1).dropna()
        y_train, X_train = combined['open'], combined[exog_features]

        X_pred = df.loc[[test_date], exog_features].apply(pd.to_numeric, errors='coerce')
        if X_pred.isnull().values.any():
            continue

        # ARIMA model
        model_arima = SARIMAX(endog=y_train, exog=X_train, order=(5, 1, 0),
                              enforce_stationarity=False, enforce_invertibility=False)
        result_arima = model_arima.fit(disp=False)
        pred_arima = result_arima.predict(start=len(y_train), end=len(y_train), exog=X_pred).iloc[0]

        # XGBoost model
        model_xgb = XGBRegressor(n_estimators=100, learning_rate=0.1)
        model_xgb.fit(X_train, y_train)
        pred_xgb = model_xgb.predict(X_pred)[0]

        # Final ensemble prediction
        pred_final = arima_weight * pred_arima + xgb_weight * pred_xgb
        y_true = df.loc[test_date, 'open']

        predictions_arima.append(pred_arima)
        predictions_xgb.append(pred_xgb)
        predictions_ensemble.append(pred_final)
        actuals.append(y_true)
        dates.append(test_date)

    if predictions_ensemble:
        st.markdown("### ✅ Features Used in This Model:")
        feature_source_map = {
            'open': "from OHLCV",
            'close': "from OHLCV",
            'high': "from OHLCV",
            'low': "from OHLCV",
            'volume': "from OHLCV",
            gt_col_name: "from Google Trend"
        }
        used_feature_labels = [f"**{f}** ({feature_source_map.get(f, 'unknown source')})" for f in selected_features]
        st.markdown("• " + "<br>• ".join(used_feature_labels), unsafe_allow_html=True)

        result_df = pd.DataFrame({
            'Date': dates,
            'Actual': actuals,
            'ARIMA': predictions_arima,
            'XGBoost': predictions_xgb,
            'Ensemble': predictions_ensemble
        }).set_index('Date')

        result_df = result_df.apply(pd.to_numeric, errors='coerce').dropna()

        fig, ax = plt.subplots()
        result_df[['Actual', 'Ensemble']].plot(marker='o', ax=ax)
        for i, row in result_df.iterrows():
            ax.annotate(f"{row['Actual']:.2f}", (i, row['Actual']), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)
            ax.annotate(f"{row['Ensemble']:.2f}", (i, row['Ensemble']), textcoords="offset points", xytext=(0, -15), ha='center', fontsize=8)
        plt.title("Actual vs Ensemble Predicted 'Open' Prices (Last 5 Days)")
        plt.ylabel("Stock Price")
        plt.grid(True)
        st.pyplot(fig)

        rmse = np.sqrt(mean_squared_error(result_df['Actual'], result_df['Ensemble']))
        st.markdown(f"### 📉 Ensemble RMSE: `{rmse:.4f}`")
    else:
        st.warning("Prediction failed for some days due to missing values or model issues.")
else:
    st.info("Please upload both datasets and select at least one feature.")
