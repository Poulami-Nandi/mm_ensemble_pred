# Multimodal Stock Price Prediction: Market + Trend Data

This project combines traditional stock market data with online search trends to forecast short-term stock price movements. It uses a blend of statistical, deep learning, and tree-based models (ARIMA, BiLSTM, XGBoost), and integrates model explainability using SHAP. The entire solution is built with a Streamlit dashboard that allows users to interactively explore model predictions.

## Project Structure
```bash
mm_ensemble_pred/
├── README.md
├── streamlit_app.py
├── requirements.txt
├── data/
│ ├── MSFT_OHLCV.csv
│ ├── TSLA_OHLCV.csv
│ ├── MSFT_trends.csv
│ └── TSLA_trends.csv
├── models/
│ ├── lstm_model.py
│ ├── arima_model.py
│ └── xgboost_model.py
├── utils/
│ ├── data_loader.py
│ ├── features.py
│ └── shap_explainer.py
└── notebooks/
└── EDA_and_Dev.ipynb
```
markdown
Copy
Edit

## Features

- **Multimodal Inputs**: Combines OHLCV stock data, Google Trends data, and technical indicators like RSI, EMA, MACD.
- **User-Configurable Input Selection**: Users can choose which inputs to include in the model training via checkboxes.
- **Ensemble Modeling**: Combine ARIMA, BiLSTM, and XGBoost outputs using custom weightings set via user input.
- **Model Explainability**: SHAP analysis shows which factors most influenced the predicted stock price movements.
- **Streamlit Interface**: A simple UI to run the entire pipeline, visualize results, and test different model combinations.

## How to Run

1. Clone this repo:
```bash
git clone https://github.com/yourusername/multimodal-stock-prediction.git
cd multimodal-stock-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Launch the app:
```bash
streamlit run streamlit_app.py
```
4. Use the dropdown to select the stock (MSFT or TSLA), choose which input types to use, and assign ensemble weights to the 3 models.

## Model Overview

- **ARIMA**: Captures seasonality and trends from historical prices.
- **BiLSTM**: Learns complex patterns and turning points in time-series data.
- **XGBoost**: Handles structured data combining trends, technical indicators, and search volume signals.

Each model is trained independently, and their predictions are combined according to user-specified weights.

## Explainability

SHAP (SHapley Additive exPlanations) is used to interpret predictions. The dashboard highlights which features had the most influence on the price movement prediction (e.g., spike in search interest, price momentum).

## Example Use Case

- Select **Tesla**
- Choose inputs: `OHLCV`, `Google Trends`, `RSI`, `MACD`
- Set ensemble weights: `BiLSTM: 0.4`, `XGBoost: 0.4`, `ARIMA: 0.2`
- View:
- Model predictions vs actual prices for last 7 days
- SHAP summary and reasoning
- Performance metrics

## License

MIT License

---

This project is designed for finance researchers, data scientists, and practitioners interested in combining behavioral and market signals for stock forecasting.
