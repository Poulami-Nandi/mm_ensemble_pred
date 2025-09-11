# Multimodal Price Forecast with Ensemble ML model (SBUX & PFE)

This repo is a self-contained demo that forecasts daily **closing prices** for two well-known tickers **Starbucks (SBUX)** and **Pfizer (PFE)** by combining multiple free, programmatic data sources.  
It uses a **multimodal ensemble**: a tree-based supervised model + a simple ARIMA time-series model, with **auto-weighted blending** to minimize validation RMSE.

The repo ships with **one year of input data already saved** under `data/`, so **no re-download is needed** to run or reproduce the demo.

---

## Contents of this project

- **Inputs (all free & programmatic; already included in `data/`)**
  - **OHLCV** via `yfinance`: the core price/volume signal used to build returns, moving averages, volatility.
  - **Google Trends** via `pytrends`: a public-attention proxy at daily granularity.
  - **GDELT News**: daily headline **volume** and **average tone** (coarse sentiment proxy).
  - **SEC EDGAR Company Facts** (XBRL JSON): slow-moving fundamentals (e.g., revenue/margins).
- **Model**
  - **Supervised track**: **XGBoost** (falls back to scikit-learn GBM if XGBoost is not available).
  - **Time-series track**: **ARIMA** (tries `pmdarima`, falls back to `statsmodels`).
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

---

## Notes on reproducibility

- Inputs are frozen in `data/` for this project. You can re-run ingestion scripts if you want, but unput data are not required for the plots above.
- If `pmdarima` is not available in your environment, ARIMA falls back to `statsmodels`.
- Python 3.11/3.12 is recommended for smooth installs.

---

## **Contact & Contributions**
Found this project useful? Feel free to ⭐ star this repo and contribute!  
**Author**: [Dr. Poulami Nandi](https://www.linkedin.com/in/poulami-nandi-a8a12917b/)  
<img src="https://github.com/Poulami-Nandi/IV_surface_analyzer/raw/main/images/own/own_image.jpg" alt="Profile" width="150"/>  
Physicist · Quant Researcher · Data Scientist  
[University of Pennsylvania](https://live-sas-physics.pantheon.sas.upenn.edu/people/poulami-nandi) | [IIT Kanpur](https://www.iitk.ac.in/) | [TU Wien](http://www.itp.tuwien.ac.at/CPT/index.htm?date=201838&cats=xbrbknmztwd)

📧 [nandi.poulami91@gmail.com](mailto:nandi.poulami91@gmail.com),    
🔗 [LinkedIn](https://www.linkedin.com/in/poulami-nandi-a8a12917b/) • [GitHub](https://github.com/Poulami-Nandi) • [Google Scholar](https://scholar.google.co.in/citations?user=bOYJeAYAAAAJ&hl=en)  


---

## License

This demo is for interview/portfolio purposes. Data comes from public, free sources for educational use.
