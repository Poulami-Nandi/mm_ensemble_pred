import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Google Trends column name (you can change this if needed)
gt_col_name = "Microsoft stock"

st.title("📈 ARIMAX Stock Price Predictor")
st.write("This app uses ARIMAX to forecast next-day stock **Open** price using selected historical features.")

# Upload files
ohlcv_file = st.file_uploader("Upload OHLCV CSV file", type="csv")
gt_file = st.file_uploader("Upload Google Trends CSV file", type="csv")

# Feature selection
all_features = ['open', 'close', 'high', 'low', 'volume', gt_col_name]
default_features = ['close', 'volume', gt_col_name]
selected_features = st.multiselect("Select features to use for prediction", all_features, default=default_features)

if ohlcv_file and gt_file and selected_features:
    # Read and merge data
    ohlcv_df = pd.read_csv(ohlcv_file, parse_dates=['date']).set_index('date')
    gt_df = pd.read_csv(gt_file, parse_dates=['Day']).set_index('Day')
    
    df = pd.merge(ohlcv_df, gt_df, left_index=True, right_index=True, how='inner')
    df = df[[*ohlcv_df.columns, gt_col_name]].dropna()

    predictions, actuals, dates = [], [], []

    # Rolling prediction for last 5 days
    for i in range(5, 0, -1):
        train = df.iloc[:-i].copy()
        test_date = df.iloc[-i:].index[0]

        # Target variable
        y_train = pd.to_numeric(train['open'], errors='coerce')

        # Exogenous variables
        X_train = train[selected_features].apply(pd.to_numeric, errors='coerce')
        combined = pd.concat([y_train, X_train], axis=1).dropna()
        if combined.empty:
            continue  # skip if no valid training data
        
        y_train = combined['open']
        X_train = combined[selected_features]

        # Exogenous input for prediction
        X_pred = df.loc[[test_date], selected_features].apply(pd.to_numeric, errors='coerce')
        if X_pred.isnull().values.any():
            continue  # skip if any missing value in prediction inputs

        try:
            model = SARIMAX(endog=y_train,
                            exog=X_train,
                            order=(5, 1, 0),
                            enforce_stationarity=False,
                            enforce_invertibility=False)
            result = model.fit(disp=False)

            y_pred = result.predict(start=len(y_train), end=len(y_train), exog=X_pred).iloc[0]
            y_true = df.loc[test_date, 'open']

            predictions.append(y_pred)
            actuals.append(y_true)
            dates.append(test_date)
        except Exception as e:
            st.warning(f"Model failed on {test_date.date()}: {e}")
            continue

    # Plot
    if predictions:
        result_df = pd.DataFrame({
            'Date': dates,
            'Actual': actuals,
            'Predicted': predictions
        }).set_index('Date')

        # Convert to float to avoid format error in annotation
        result_df['Actual'] = pd.to_numeric(result_df['Actual'], errors='coerce')
        result_df['Predicted'] = pd.to_numeric(result_df['Predicted'], errors='coerce')

        fig, ax = plt.subplots()
        result_df.plot(marker='o', ax=ax)

        for i, row in result_df.iterrows():
            if not pd.isna(row['Actual']):
                ax.annotate(f"{row['Actual']:.2f}", (i, row['Actual']),
                            textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)
            if not pd.isna(row['Predicted']):
                ax.annotate(f"{row['Predicted']:.2f}", (i, row['Predicted']),
                            textcoords="offset points", xytext=(0, -15), ha='center', fontsize=8)

        plt.title("Actual vs Predicted 'Open' Prices (Last 5 Trading Days)")
        plt.ylabel("Stock Price")
        plt.grid(True)
        st.pyplot(fig)
    else:
        st.warning("Prediction failed for some days due to missing values or model issues.")
