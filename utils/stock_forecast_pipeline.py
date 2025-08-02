
import os
import datetime
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings
import traceback

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

class StockForecastPipeline:
    def __init__(self, ticker, start_date, end_date, arima_weight=0.5, data_dir="data"):
        self.ticker = ticker.upper()
        self.start_date = pd.to_datetime(start_date).date()
        self.end_date = pd.to_datetime(end_date).date()
        self.data_dir = data_dir
        self.gt_col = f"{self.ticker}_trend"
        self.ohlcv_path = os.path.join(data_dir, f"{self.ticker}_OHLCV_{self.start_date:%d%b%Y}_{self.end_date:%d%b%Y}.csv")
        self.gt_path = os.path.join(data_dir, f"{self.ticker}_GT_{self.start_date:%d%b%Y}_{self.end_date:%d%b%Y}.csv")
        self.arima_order = (5, 1, 0)
        self.arima_weight = arima_weight
        self.xgb_weight = 1 - arima_weight
        os.makedirs(self.data_dir, exist_ok=True)

    def download_ohlcv_data(self):
        if os.path.exists(self.ohlcv_path):
            print(f"Using cached OHLCV file: {self.ohlcv_path}")
        else:
            print(f"Downloading OHLCV data for {self.ticker}...")
            df = yf.download(self.ticker, start=self.start_date, end=self.end_date)

            if len(df) < 1250:
                raise ValueError(f"Not enough data for {self.ticker}. Found {len(df)} rows.")

            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            df = df.dropna(how='any')
            df.index = pd.to_datetime(df.index)
            df.index.name = 'Date'
            df.to_csv(self.ohlcv_path)

    def generate_google_trend_data(self, ohlcv_df):
        if os.path.exists(self.gt_path):
            print(f"Using cached Google Trend file: {self.gt_path}")
            return

        print(f"Generating Google Trends data for {self.ticker}...")
        dates = pd.to_datetime(ohlcv_df.index, errors='coerce').dropna()
        gt_df = pd.DataFrame(index=dates)
        np.random.seed(42)
        gt_df[self.gt_col] = np.random.randint(10, 300, size=len(dates))
        gt_df.index.name = 'Day'
        gt_df.to_csv(self.gt_path)
        print(f"Saved Google Trends data to {self.gt_path}")

    def derive_features(self):
        df_ohlcv = pd.read_csv(self.ohlcv_path, index_col=0, parse_dates=True)
        df_ohlcv = df_ohlcv[['Open', 'Close', 'Volume']].copy()
        df_ohlcv.columns = df_ohlcv.columns.str.lower()
        for col in df_ohlcv.columns:
            df_ohlcv[col] = pd.to_numeric(df_ohlcv[col], errors='coerce')

        df_ohlcv['return_1d'] = df_ohlcv['close'].pct_change(fill_method=None)
        df_ohlcv['ema_20'] = df_ohlcv['close'].ewm(span=20).mean()
        df_ohlcv['volatility_10d'] = df_ohlcv['return_1d'].rolling(10).std()
        df_ohlcv = df_ohlcv[['return_1d', 'ema_20', 'volatility_10d']]
        df_ohlcv.to_csv(os.path.join(self.data_dir, f"derived_{os.path.basename(self.ohlcv_path)}"))

        df_gt = pd.read_csv(self.gt_path, parse_dates=['Day']).set_index('Day')
        df_gt[self.gt_col] = pd.to_numeric(df_gt[self.gt_col], errors='coerce')
        df_gt['trend_7d_ma'] = df_gt[self.gt_col].rolling(7).mean()
        df_gt['trend_rolling_max_50d'] = df_gt[self.gt_col].rolling(50).max()
        df_gt['trend_return'] = df_gt[self.gt_col].pct_change(fill_method=None)
        df_gt = df_gt[['trend_7d_ma', 'trend_rolling_max_50d', 'trend_return']]
        df_gt.to_csv(os.path.join(self.data_dir, f"derived_{os.path.basename(self.gt_path)}"))

    def load_data(self):
        ohlcv_df = pd.read_csv(self.ohlcv_path, index_col=0, parse_dates=True)
        ohlcv_df.index = pd.to_datetime(ohlcv_df.index, errors='coerce')
        ohlcv_df = ohlcv_df[~ohlcv_df.index.isna()]
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            ohlcv_df[col] = pd.to_numeric(ohlcv_df[col], errors='coerce')

        df_ohlcv_der = pd.read_csv(os.path.join(self.data_dir, f"derived_{os.path.basename(self.ohlcv_path)}"),
                                   index_col=0, parse_dates=True)
        df_ohlcv_der.index = pd.to_datetime(df_ohlcv_der.index, errors='coerce')
        df_ohlcv_der = df_ohlcv_der[~df_ohlcv_der.index.isna()]

        df_gt_der = pd.read_csv(os.path.join(self.data_dir, f"derived_{os.path.basename(self.gt_path)}"),
                                index_col=0, parse_dates=True)
        df_gt_der.index = pd.to_datetime(df_gt_der.index, errors='coerce')
        df_gt_der = df_gt_der[~df_gt_der.index.isna()]

        common_index = ohlcv_df.index.intersection(df_ohlcv_der.index).intersection(df_gt_der.index)
        ohlcv_df = ohlcv_df.loc[common_index]
        df_ohlcv_der = df_ohlcv_der.loc[common_index]
        df_gt_der = df_gt_der.loc[common_index]

        df = pd.concat([ohlcv_df[['Open', 'High', 'Low', 'Close', 'Volume']].rename(columns=str.lower),
                        df_ohlcv_der, df_gt_der], axis=1).dropna()

        try:
            inferred_freq = pd.infer_freq(df.index)
            if inferred_freq:
                df.index.freq = inferred_freq
        except Exception:
            pass

        return df

    def train_test_split_rolling(self, df, selected_features, target="open", n_forecasts=30):
        rolling_data = []
        for i in range(n_forecasts, 0, -1):
            train = df.iloc[:-i]
            test_date = df.iloc[-i:].index[0]
            y_train = train[target]
            X_train = train[selected_features]
            X_test = df.loc[[test_date], selected_features]
            actual = df.loc[test_date, target]
            rolling_data.append((X_train, y_train, X_test, actual, test_date))
        return rolling_data


    def predict(self, data):
        records = []
        arima_coefs_list = []
        xgb_feat_importances = []
        feat_names = None

        for X_train, y_train, X_test, actual, test_date in data:
            try:
                X_train = X_train.dropna(axis=1)  # drop columns with all NaNs
                X_train = X_train.fillna(0)      # fill any remaining NaNs
                y_train.index.freq = pd.infer_freq(y_train.index)
                print(f"Training ARIMA for {test_date} and X_train=\n{X_train.head(2)}")
                arima_model = SARIMAX(endog=y_train, exog=X_train, order=self.arima_order,
                                      enforce_stationarity=False, enforce_invertibility=False)
                arima_result = arima_model.fit(disp=False)
                arima_pred = arima_result.predict(start=len(y_train), end=len(y_train), exog=X_test).values[0]
                arima_coefs = arima_result.params.filter(like="x")
                arima_coefs_list.append(arima_coefs.abs())
            except Exception as e:
                print(f"[WARN] ARIMA failed at {test_date}: {e}")
                arima_pred = np.nan
                arima_coefs_list.append(pd.Series(np.nan, index=X_train.columns))

            try:
                xgb_model = XGBRegressor()
                xgb_model.fit(X_train, y_train)
                xgb_pred = xgb_model.predict(X_test)[0]
                xgb_feat_importances.append(xgb_model.feature_importances_)
                feat_names = X_train.columns
            except Exception as e:
                print(f"[ERROR] XGBoost failed at {test_date}: {e}")
                xgb_pred = np.nan
                xgb_feat_importances.append([np.nan] * X_train.shape[1])

            if np.isnan(arima_pred):
                ensemble_pred = xgb_pred
            elif np.isnan(xgb_pred):
                ensemble_pred = arima_pred
            else:
                ensemble_pred = self.arima_weight * arima_pred + self.xgb_weight * xgb_pred

            records.append({
                "date": test_date,
                "actual": actual,
                "arima_pred": arima_pred,
                "xgb_pred": xgb_pred,
                "ensemble_pred": ensemble_pred
            })

        self._arima_feat_df = pd.DataFrame(arima_coefs_list).fillna(0)
        self._xgb_feat_df = pd.DataFrame(xgb_feat_importances, columns=feat_names).fillna(0)

        return pd.DataFrame(records).set_index("date")

    def feature_importance(self):
        arima_importance = self._arima_feat_df.mean()
        arima_importance = 100 * arima_importance / arima_importance.sum()

        xgb_importance = self._xgb_feat_df.mean()
        xgb_importance = 100 * xgb_importance / xgb_importance.sum()

        arima_importance = arima_importance.fillna(0)
        xgb_importance = xgb_importance.fillna(0)
        ensemble_importance = (arima_importance * self.arima_weight) + (xgb_importance * self.xgb_weight)
        ensemble_importance = 100 * ensemble_importance / ensemble_importance.sum()

        print("\nFeature Importance (%):")
        print("ARIMA:\n", arima_importance.round(2))
        print("XGBoost:\n", xgb_importance.round(2))
        print("Ensemble:\n", ensemble_importance.round(2))

        return pd.DataFrame({
            "ARIMA (%)": arima_importance.round(2),
            "XGBoost (%)": xgb_importance.round(2),
            "Ensemble (%)": ensemble_importance.round(2)
        })


    def evaluate(self, df_result):
        print("Evaluation results:")
        rmse_arima = mean_squared_error(df_result["actual"], df_result["arima_pred"])
        rmse_xgb = mean_squared_error(df_result["actual"], df_result["xgb_pred"])
        rmse_ensemble = mean_squared_error(df_result["actual"], df_result["ensemble_pred"])
        print(f"RMSE (ARIMA):    {rmse_arima:.4f}")
        print(f"RMSE (XGBoost):  {rmse_xgb:.4f}")
        print(f"RMSE (Ensemble): {rmse_ensemble:.4f}")

        return rmse_arima, rmse_xgb, rmse_ensemble

    def plot(self, result_df):
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(result_df.index, result_df["actual"], label="Actual", color="black", linewidth=2)
        ax.plot(result_df.index, result_df["arima_pred"], label="ARIMA", linestyle="--")
        ax.plot(result_df.index, result_df["xgb_pred"], label="XGBoost", linestyle="-.", color="orangered")
        ax.plot(result_df.index, result_df["ensemble_pred"], label="Ensemble", color="green")

        ax.set_title(f"{self.ticker} Price Forecast (Last {len(result_df)} Trading Days)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(True)

        # Fix x-axis: show only one tick per day
        ax.set_xticks(result_df.index)
        ax.set_xticklabels([d.strftime("%Y-%m-%d") for d in result_df.index], rotation=45)

        plt.tight_layout()
        plt.show()


    def plot_feature_importance(self, df, title="Feature Importance Comparison"):
        ax = df.plot(kind='bar', figsize=(12, 6))
        plt.title(title)
        plt.ylabel("Importance (%)")
        plt.xlabel("Features")
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.legend(title="Model")
        plt.show()

