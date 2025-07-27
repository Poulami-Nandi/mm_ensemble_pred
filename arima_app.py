import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)

gt_col_name = "Microsoft stock"

st.title("📈 ARIMAX Stock Price Predictor")
st.write("This app uses ARIMAX to forecast next-day stock **Open** price using selected historical features.")

# Upload files
ohlcv_file = st.file_uploader("Upload OHLCV CSV file", type="csv")
gt_file = st.file_uploader("Upload Google Trends CSV file", type="csv")

# Feature selection
all_features = ['open', 'close', 'high', 'low', 'volume', gt_col_name]
selected_features = st.multiselect("Select features to use for prediction", all_features, default=['close', 'volume', gt_col_name])

if ohlcv_file and gt_file and selected_features:
    # Read and merge data
    ohlcv_df = pd.read_csv(ohlcv_file, parse_dates=['date']).set_index('date')
    gt_df = pd.read_csv(gt_file, parse_dates=['Day']).set_index('Day')
    df = pd.merge(ohlcv_df, gt_df, left_index=True, right_index=True, how='inner')
    df = df[[*ohlcv_df.columns, gt_col_name]].dropna()

    predictions, actuals, dates = [], [], []

    for i in range(5, 0, -1):
        train = df.iloc[:-i].copy()
        test_date = df.iloc[-i:].index[0]

        y_train = pd.to_numeric(train['open'], errors='coerce')
        X_train = train[selected_features].apply(pd.to_numeric, errors='coerce')
        combined = pd.concat([y_train, X_train], axis=1).dropna()
        y_train, X_train = combined['open'], combined[selected_features]

        X_pred = df.loc[[test_date], selected_features].apply(pd.to_numeric, errors='coerce')
        if X_pred.isnull().values.any():
            continue

        model = SARIMAX(endog=y_train, exog=X_train, order=(5, 1, 0),
                        enforce_stationarity=False, enforce_invertibility=False)
        result = model.fit(disp=False)

        y_pred = result.predict(start=len(y_train), end=len(y_train), exog=X_pred).iloc[0]
        y_true = df.loc[test_date, 'open']

        predictions.append(y_pred)
        actuals.append(y_true)
        dates.append(test_date)

    # Show selected features with source
    if predictions:
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

        # Create result dataframe
        result_df = pd.DataFrame({'Date': dates, 'Actual': actuals, 'Predicted': predictions}).set_index('Date')
        result_df['Actual'] = pd.to_numeric(result_df['Actual'], errors='coerce')
        result_df['Predicted'] = pd.to_numeric(result_df['Predicted'], errors='coerce')
        result_df = result_df.dropna()

        # Plot Actual vs Predicted
        fig, ax = plt.subplots()
        result_df.plot(marker='o', ax=ax)
        for i, row in result_df.iterrows():
            ax.annotate(f"{row['Actual']:.2f}", (i, row['Actual']), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)
            ax.annotate(f"{row['Predicted']:.2f}", (i, row['Predicted']), textcoords="offset points", xytext=(0, -15), ha='center', fontsize=8)
        plt.title("Actual vs Predicted 'Open' Prices (Last 5 Trading Days)")
        plt.ylabel("Stock Price")
        plt.grid(True)
        st.pyplot(fig)

        # RMSE
        rmse = mean_squared_error(result_df['Actual'], result_df['Predicted'], squared=False)
        st.markdown(f"### 📉 RMSE (Root Mean Squared Error): `{rmse:.4f}`")

    else:
        st.warning("Prediction failed for some days due to missing values or model issues.")
else:
    st.info("Please upload both datasets and select at least one feature.")
