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

        arima_rmse = np.sqrt(mean_squared_error(results_df['Actual'], results_df['ARIMA']))
        xgb_rmse = np.sqrt(mean_squared_error(results_df['Actual'], results_df['XGBoost']))
        ensemble_rmse = np.sqrt(mean_squared_error(results_df['Actual'], results_df['Ensemble']))

        st.subheader("📉 Forecast Plot (Last 5 Trading Days)")
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(results_df.index, results_df['Actual'], label='Actual', marker='o', linewidth=2, color='black')
        ax.plot(results_df.index, results_df['ARIMA'], label='ARIMA', marker='o', linestyle='--')
        ax.plot(results_df.index, results_df['XGBoost'], label='XGBoost', marker='o', linestyle='--')
        ax.plot(results_df.index, results_df['Ensemble'], label='Ensemble', marker='o', linewidth=2)

        for i, row in results_df.iterrows():
            try:
                ax.annotate(f"{row['Actual']:.2f}", (row.name, row['Actual']), textcoords="offset points",
                            xytext=(0, 10), ha='center', fontsize=8, color='black')
                ax.annotate(f"{row['Ensemble']:.2f}", (row.name, row['Ensemble']), textcoords="offset points",
                            xytext=(0, -15), ha='center', fontsize=8, color='green')
            except Exception:
                continue

        ax.set_title("Actual vs Predicted 'Open' Prices")
        ax.set_ylabel("Stock Price")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
        ax.set_xticks(results_df.index)
        ax.set_xticklabels(results_df.index.strftime('%Y-%m-%d'), rotation=45, ha='right')
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("📊 RMSE Comparison")
        st.markdown(f"- **ARIMA RMSE:** {arima_rmse:.4f}")
        st.markdown(f"- **XGBoost RMSE:** {xgb_rmse:.4f}")
        st.markdown(f"- **Ensemble RMSE:** {ensemble_rmse:.4f}")
    else:
        st.warning("Prediction failed. Check for missing or inconsistent data.")
else:
    st.info("Upload data, select features, and click **Run Prediction**.")
