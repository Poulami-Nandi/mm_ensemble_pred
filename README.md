# Multimodal Stock Price Prediction: Market + Trend Data

This project combines traditional stock market data with online search trends to forecast short-term stock price movements. It uses a blend of statistical, deep learning, and tree-based models (ARIMA, BiLSTM, XGBoost), and integrates model explainability using SHAP. The entire solution is built with a Streamlit dashboard that allows users to interactively explore model predictions.

## Project Structure
```bash
mm_ensemble_pred/
├── data/                         # saved input snapshots (no re-download)
│   ├── PFE/
│   │   ├── prices.parquet
│   │   ├── trends.parquet
│   │   ├── gdelt_daily.csv
│   │   ├── facts.json
│   │   └── dataset.parquet
│   └── SBUX/                     # same structure as PFE
│       ├── prices.parquet
│       ├── trends.parquet
│       ├── gdelt_daily.csv
│       ├── facts.json
│       └── dataset.parquet
├── outputs/
│   └── last5/                    # precomputed last-5-day artifacts
│       ├── PFE/
│       │   ├── actual_vs_pred_all_inputs_PRICE.png
│       │   ├── actual_vs_pred_ohlcv_only_PRICE.png
│       │   ├── compare_all_vs_ohlcv_PRICE.png
│       │   ├── pred_last5_all_inputs.csv
│       │   ├── pred_last5_ohlcv_only.csv
│       │   ├── metrics_all_inputs.json
│       │   └── metrics_ohlcv_only.json
│       ├── SBUX/                 # same structure as PFE
│       │   ├── actual_vs_pred_all_inputs_PRICE.png
│       │   ├── actual_vs_pred_ohlcv_only_PRICE.png
│       │   ├── compare_all_vs_ohlcv_PRICE.png
│       │   ├── pred_last5_all_inputs.csv
│       │   ├── pred_last5_ohlcv_only.csv
│       │   ├── metrics_all_inputs.json
│       │   └── metrics_ohlcv_only.json
│       └── summary.json          # compact run summary (RMSE, pointers)
├── src/
│   └── mm_ensemble/
│       ├── __init__.py
│       ├── streamlit_app.py      # static viewer (shows saved artifacts only)
│       ├── trainer.py            # XGB/GBM + ARIMA + auto-blend weights
│       ├── build_dataset.py      # assembles features from saved inputs
│       ├── last5_ensemble_plots.py
│       ├── inference.py
│       ├── backtest.py
│       ├── plot_actual_vs_pred.py
│       ├── build_fundamentals.py
│       ├── data_ingest.py
│       └── utils/
│           ├── __init__.py
│           └── paths.py          # repo-relative path helpers
├── pyproject.toml
├── README.md
├── .gitattributes
└── .gitignore

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


# mm_ensemble â€” Multimodal Price Forecast Demo (SBUX & PFE)

This repo is a small, self-contained demo that forecasts daily **closing prices** for two well-known tickers â€” **Starbucks (SBUX)** and **Pfizer (PFE)** â€” by combining multiple free, programmatic data sources.  
It uses a **multimodal ensemble**: a tree-based supervised model + a simple ARIMA time-series model, with **auto-weighted blending** to minimize validation RMSE.

The repo ships with **one year of input data already saved** under `data/`, so **no re-download is needed** to run or reproduce the demo.

---

## Whatâ€™s inside (in plain English)

- **Inputs (all free & programmatic; already included in `data/`)**
  - **OHLCV** via `yfinance`: the core price/volume signal used to build returns, moving averages, volatility.
  - **Google Trends** via `pytrends`: a public-attention proxy at daily granularity.
  - **GDELT News**: daily headline **volume** and **average tone** (coarse sentiment proxy).
  - **SEC EDGAR Company Facts** (XBRL JSON): slow-moving fundamentals (e.g., revenue/margins).
- **Model**
  - **Supervised track**: **XGBoost** (falls back to scikit-learn GBM if XGBoost isnâ€™t available).
  - **Time-series track**: **ARIMA** (tries `pmdarima`, falls back to `statsmodels`, else naive).
  - **Ensemble**: learn a single **blend weight** on the validation split that minimizes RMSE; clamp to [0, 1].  
    Final prediction = `w * (XGB/GBM) + (1 - w) * (ARIMA)`.
- **Last-5-day demo**
  - Train on ~last 1y (minus 5 trading days), predict the **last 5 trading days**.
  - Do this **twice** per ticker:
    1) **All inputs** (OHLCV + Trends + News + Fundamentals)  
    2) **OHLCV only**  
  - Save CSV/JSON/PNG artifacts under `outputs/last5/<TICKER>/`.

---

## Quick start (local)

```bash
# 1) clone
git clone https://github.com/Poulami-Nandi/mm_ensemble_pred.git
cd mm_ensemble_pred

# 2) (optional but recommended) create a virtual env for Python 3.9â€“3.12
# python -m venv .venv && source .venv/bin/activate

# 3) install in editable mode
pip install -e .

# 4) (optional) run the static Streamlit viewer for the saved artifacts
streamlit run src/mm_ensemble/streamlit_app.py
```

Note: The app does not re-download inputs or re-train. It simply displays the precomputed plots and metrics already in the repo.

---

## Last-5-day plots (from this repo)

These images are committed in `outputs/last5/...` and linked directly below (via raw GitHub links).

### Pfizer (PFE)

**a) OHLCV vs Actual**  
![PFE OHLCV vs Actual](https://raw.githubusercontent.com/Poulami-Nandi/mm_ensemble_pred/main/outputs/last5/PFE/actual_vs_pred_ohlcv_only_PRICE.png)

**b) All inputs vs Actual**  
![PFE All inputs vs Actual](https://raw.githubusercontent.com/Poulami-Nandi/mm_ensemble_pred/main/outputs/last5/PFE/actual_vs_pred_all_inputs_PRICE.png)

**c) OHLCV vs All inputs vs Actual**  
![PFE Compare](https://raw.githubusercontent.com/Poulami-Nandi/mm_ensemble_pred/main/outputs/last5/PFE/compare_all_vs_ohlcv_PRICE.png)

### Starbucks (SBUX)

**a) OHLCV vs Actual**  
![SBUX OHLCV vs Actual](https://raw.githubusercontent.com/Poulami-Nandi/mm_ensemble_pred/main/outputs/last5/SBUX/actual_vs_pred_ohlcv_only_PRICE.png)

**b) All inputs vs Actual**  
![SBUX All inputs vs Actual](https://raw.githubusercontent.com/Poulami-Nandi/mm_ensemble_pred/main/outputs/last5/SBUX/actual_vs_pred_all_inputs_PRICE.png)

**c) OHLCV vs All inputs vs Actual**  
![SBUX Compare](https://raw.githubusercontent.com/Poulami-Nandi/mm_ensemble_pred/main/outputs/last5/SBUX/compare_all_vs_ohlcv_PRICE.png)

A compact run summary lives here:  
`outputs/last5/summary.json` (raw):  
https://raw.githubusercontent.com/Poulami-Nandi/mm_ensemble_pred/main/outputs/last5/summary.json

---

## How it works (short)

1. **Dataset build**  
   Combine saved inputs for the last ~1 year into supervised features (technical indicators from OHLCV and simple transforms for Trends/News/Facts).
2. **Train/Val/Test split**  
   - Train: most of the last year  
   - Val: small tail chunk from the training period  
   - Test: **last 5 trading days** (held out)
3. **Fit two models**  
   - XGB/GBM on the features  
   - ARIMA on the target series
4. **Auto-weighting**  
   Solve for the blend weight `w` on the **validation** set that minimizes RMSE; clamp to [0,1].
5. **Predict & save**  
   Predict the last 5 days, write CSV/JSON/PNG artifacts under `outputs/last5/<TICKER>/`.

---

## Backtests & artifacts

- **Per-ticker artifacts** (CSV/JSON/plots):  
  `outputs/last5/SBUX/` and `outputs/last5/PFE/`
- **Run summary** (RMSE, pointers):  
  `outputs/last5/summary.json`

If you add more runs or tickers, keep the same folder structure so the Streamlit app can pick them up automatically.

---

## Project layout (relevant bits)

```
.
â”œâ”€â”€ data/                     # saved, one-time input snapshots (no re-download)
â”‚   â”œâ”€â”€ SBUX/                 # prices, trends, news, facts
â”‚   â””â”€â”€ PFE/
â”œâ”€â”€ outputs/
â”‚   â””â”€â”€ last5/
â”‚       â”œâ”€â”€ SBUX/             # last-5 predictions, metrics, plots
â”‚       â””â”€â”€ PFE/
â”œâ”€â”€ src/
â”‚   â””â”€â”€ mm_ensemble/
â”‚       â”œâ”€â”€ streamlit_app.py  # static viewer for saved artifacts
â”‚       â”œâ”€â”€ trainer.py        # training (XGB/GBM + ARIMA + auto blend)
â”‚       â”œâ”€â”€ last5_ensemble_plots.py
â”‚       â”œâ”€â”€ build_dataset.py  # feature assembly (uses saved inputs)
â”‚       â””â”€â”€ utils/paths.py    # path helpers
â”œâ”€â”€ pyproject.toml
â””â”€â”€ README.md
```

---

## Notes on reproducibility

- Inputs are frozen in `data/` for this demo. You can re-run ingestion scripts if you want, but itâ€™s not required for the plots above.
- If `pmdarima` isnâ€™t available in your environment, ARIMA falls back to `statsmodels`, then to a naive baseline â€” the ensemble still works.
- Python 3.9â€“3.12 is recommended for smooth installs (especially around `pmdarima` wheels).

---

## License

This demo is for interview/portfolio purposes. Data comes from public, free sources for educational use.
