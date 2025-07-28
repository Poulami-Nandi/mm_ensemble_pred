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

st.title("📈 Ensemble Stock Price Predictor (ARIMA + XGBoost)")
st.write("This app forecasts next-day stock **Open** price using an ensemble of ARIMA and XGBoost models.")

# Upload files
ohlcv_file = st.file_uploader("Upload OHLCV CSV file", type="csv")
gt_file = st.file_uploader("Upload Google Trends CSV file", type="csv")

# Feature selection
gt_col_name = "Microsoft stock"
all_features = ['open', 'close', 'high', 'low', 'volume', gt_col_name]
selected_features = st.multiselect("Select features to use for prediction", all_features, default=['close', 'volume', gt_col_name])

# Weight slider
arima_weight = st.slider("ARIMA Weight (%)", 0, 100, 50)
xgb_weight = 100 - arima_weight

# Trigger button
trigger = st.button("Press to Run Forecast")

if trigger and ohlcv_file and gt_file and selected_features:
    # Load and merge data
    ohlcv_df = pd.read_csv(ohlcv_file, parse_dates=['date']).set_index('date')
    gt_df = pd.read_csv(gt_file, parse_dates=['Day']).set_index('Day')
    df = pd.merge(ohlcv_df, gt_df, left_index=True, right_index=True, how='inner')
    df = df[[*ohlcv_df.columns, gt_col_name]].dropna()

    arima_preds, xgb_preds, actuals, dates = [], [], [], []

    for i in range(5, 0, -1):
        train = df.iloc[:-i].copy()
        test_date = df.iloc[-i:].index[0]

        # Prepare data
        y_train = pd.to_numeric(train['open'], errors='coerce')
        X_train = train[selected_features].apply(pd.to_numeric, errors='coerce')
        combined = pd.concat([y_train, X_train], axis=1).dropna()
        y_train, X_train = combined['open'], combined[selected_features]
        X_pred = df.loc[[test_date], selected_features].apply(pd.to_numeric, errors='coerce')

        if y_train.empty or X_train.empty or X_pred.isnull().values.any():
            continue

        # ARIMA model
        arima_model = SARIMAX(endog=y_train, exog=X_train, order=(5, 1, 0),
                              enforce_stationarity=False, enforce_invertibility=False)
        arima_result = arima_model.fit(disp=False)
        arima_pred = arima_result.predict(start=len(y_train), end=len(y_train), exog=X_pred).iloc[0]

        # XGBoost model
        xgb_model = XGBRegressor(n_estimators=100, max_depth=3)
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_pred)[0]

        actual = df.loc[test_date, 'open']
        ensemble_pred = (arima_weight / 100) * arima_pred + (xgb_weight / 100) * xgb_pred

        arima_preds.append(arima_pred)
        xgb_preds.append(xgb_pred)
        actuals.append(actual)
        dates.append(test_date)

    # Display results
    if dates:
        result_df = pd.DataFrame({
            'Date': dates,
            'Actual': actuals,
            'ARIMA': arima_preds,
            'XGBoost': xgb_preds
        }).set_index('Date')
        result_df['Ensemble'] = (arima_weight / 100) * result_df['ARIMA'] + (xgb_weight / 100) * result_df['XGBoost']

        st.markdown("### Features used for training:")
        for feat in selected_features:
            origin = "Google Trend" if feat == gt_col_name else "OHLCV"
            st.markdown(f"- **{feat}** from *{origin}*")

        # Plotting
        fig, ax = plt.subplots()
        result_df[['Actual', 'ARIMA', 'XGBoost', 'Ensemble']].plot(marker='o', ax=ax)
        for i, row in result_df.iterrows():
            ax.annotate(f"{float(row['Actual']):.2f}", (i, row['Actual']), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
            ax.annotate(f"{float(row['Ensemble']):.2f}", (i, row['Ensemble']), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=8)
        plt.title("Actual vs Predicted 'Open' Prices (Last 5 Trading Days)")
        plt.ylabel("Stock Price")
        plt.grid(True)
        st.pyplot(fig)

        # RMSEs
        rmse_arima = mean_squared_error(result_df['Actual'], result_df['ARIMA'], squared=False)
        rmse_xgb = mean_squared_error(result_df['Actual'], result_df['XGBoost'], squared=False)
        rmse_ensemble = mean_squared_error(result_df['Actual'], result_df['Ensemble'], squared=False)

        st.success(f" **RMSE (ARIMA):** {rmse_arima:.4f}")
        st.success(f" **RMSE (XGBoost):** {rmse_xgb:.4f}")
        st.success(f" **RMSE (Ensemble):** {rmse_ensemble:.4f}")
    else:
        st.warning("Prediction failed due to missing or invalid values.")
elif trigger:
    st.info("Please upload both datasets and select at least one feature.")
