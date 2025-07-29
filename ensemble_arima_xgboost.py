import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings("ignore")

gt_col_name = "Microsoft stock"

st.title("🔮 Ensemble Forecast: ARIMA + XGBoost for Stock 'Open' Price")
st.write("Predicting next-day **Open** prices using selected features from OHLCV + Google Trends via ARIMAX and XGBoost ensemble.")

# File Upload
ohlcv_file = st.file_uploader("Upload OHLCV CSV file", type="csv")
gt_file = st.file_uploader("Upload Google Trends CSV file", type="csv")

# Feature selection
all_features = ['open', 'close', 'high', 'low', 'volume', gt_col_name]
selected_features = st.multiselect("Select features for prediction", all_features, default=['close', 'volume', gt_col_name])

# Weight slider
arima_weight = st.slider("ARIMA weight in Ensemble Prediction", 0.0, 1.0, 0.5, 0.05)

# Run Button
trigger = st.button("Run Prediction")

if ohlcv_file and gt_file and selected_features and trigger:
    ohlcv_df = pd.read_csv(ohlcv_file, parse_dates=['date']).set_index('date')
    gt_df = pd.read_csv(gt_file, parse_dates=['Day']).set_index('Day')

    df = pd.merge(ohlcv_df, gt_df, left_index=True, right_index=True, how='inner')
    df = df[[*ohlcv_df.columns, gt_col_name]].dropna()

    st.subheader("🧠 Features Used")
    used_features = []
    for feat in selected_features:
        if feat in ohlcv_df.columns:
            used_features.append(f"'{feat}' from OHLCV")
        else:
            used_features.append(f"'{feat}' from Google Trend data")
    st.markdown(", ".join(used_features))

    dates, y_true_list, arima_preds, xgb_preds, ensemble_preds = [], [], [], [], []

    for i in range(5, 0, -1):
        train = df.iloc[:-i].copy()
        test_date = df.iloc[-i:].index[0]

        y_train = pd.to_numeric(train['open'], errors='coerce')
        X_train = train[selected_features].apply(pd.to_numeric, errors='coerce')
        combined = pd.concat([y_train, X_train], axis=1).dropna()
        y_train, X_train = combined['open'], combined[selected_features]

        X_test = df.loc[[test_date], selected_features].apply(pd.to_numeric, errors='coerce')
        if X_test.isnull().values.any():
            continue

        # ARIMA
        arima_model = SARIMAX(endog=y_train, exog=X_train, order=(5, 1, 0),
                              enforce_stationarity=False, enforce_invertibility=False)
        arima_result = arima_model.fit(disp=False)
        arima_pred = arima_result.predict(start=len(y_train), end=len(y_train), exog=X_test).iloc[0]

        # XGBoost
        xgb_model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, objective='reg:squarederror')
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)[0]

        actual = df.loc[test_date, 'open']

        dates.append(test_date)
        y_true_list.append(actual)
        arima_preds.append(arima_pred)
        xgb_preds.append(xgb_pred)
        ensemble_preds.append(arima_weight * arima_pred + (1 - arima_weight) * xgb_pred)

    if dates:
        results_df = pd.DataFrame({
            'Date': dates,
            'Actual': y_true_list,
            'ARIMA': arima_preds,
            'XGBoost': xgb_preds,
            'Ensemble': ensemble_preds
        }).set_index('Date')

        # Ensure numeric values for proper plotting
        for col in ['Actual', 'ARIMA', 'XGBoost', 'Ensemble']:
            results_df[col] = pd.to_numeric(results_df[col], errors='coerce')

        plot_df = results_df.dropna(subset=['Actual', 'ARIMA', 'XGBoost', 'Ensemble'])

        arima_rmse = np.sqrt(mean_squared_error(plot_df['Actual'], plot_df['ARIMA']))
        xgb_rmse = np.sqrt(mean_squared_error(plot_df['Actual'], plot_df['XGBoost']))
        ensemble_rmse = np.sqrt(mean_squared_error(plot_df['Actual'], plot_df['Ensemble']))

        # Plot
        st.subheader("📉 Forecast Plot (Last 5 Trading Days)")
        fig, ax = plt.subplots()

        plot_df[['Actual', 'ARIMA', 'XGBoost', 'Ensemble']].plot(ax=ax, marker='o')

        y_min = plot_df[['Actual', 'ARIMA', 'XGBoost', 'Ensemble']].min().min()
        y_max = plot_df[['Actual', 'ARIMA', 'XGBoost', 'Ensemble']].max().max()
        y_margin = (y_max - y_min) * 0.1
        ax.set_ylim(y_min - y_margin, y_max + y_margin)

        plt.ylabel("Stock Price")
        plt.title("Actual vs Predicted 'Open' Prices")
        plt.xticks(rotation=45)
        plt.grid(True)
        st.pyplot(fig)

        st.subheader("📊 RMSE Comparison")
        st.markdown(f"- **ARIMA RMSE:** {arima_rmse:.4f}")
        st.markdown(f"- **XGBoost RMSE:** {xgb_rmse:.4f}")
        st.markdown(f"- **Ensemble RMSE:** {ensemble_rmse:.4f}")
    else:
        st.warning("Prediction failed. Check for missing or inconsistent data.")
else:
    st.info("Upload data, select features, and click **Run Prediction**.")
