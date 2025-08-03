import gradio as gr
import pandas as pd
from utils.stock_forecast_pipeline import StockForecastPipeline

def get_feature_groups(ticker, arima_weight, start_date, end_date):
    pipeline = StockForecastPipeline(ticker, start_date, end_date, arima_weight / 100)
    pipeline.download_ohlcv_data()
    ohlcv_df = pd.read_csv(pipeline.ohlcv_path, index_col=0, parse_dates=True)
    pipeline.generate_google_trend_data(ohlcv_df)
    pipeline.derive_features()

    df = pipeline.load_data()

    # Categorize features
    all_features = df.columns.tolist()
    ohlcv = ["open", "high", "low", "close", "volume"]
    ohlcv_derived = [col for col in df.columns if col not in ohlcv and not col.startswith(ticker.lower()) and not col.startswith("trend_")]
    gt = [pipeline.gt_col] if hasattr(pipeline, 'gt_col') else [col for col in df.columns if col.lower() == f"{ticker.lower()}_trend"]
    gt_derived = [col for col in df.columns if col.startswith("trend_")]

    return ohlcv, ohlcv_derived, gt, gt_derived

def run_forecast(ticker, arima_weight, start_date, end_date,
                 selected_ohlcv, selected_ohlcv_der, selected_gt, selected_gt_der):
    selected_features = selected_ohlcv + selected_ohlcv_der + selected_gt + selected_gt_der
    pipeline = StockForecastPipeline(ticker, start_date, end_date, arima_weight / 100)
    df = pipeline.load_data()
    data = pipeline.train_test_split_rolling(df, selected_features)
    result_df = pipeline.predict(data)
    rmse_arima, rmse_xgb, rmse_ensemble = pipeline.evaluate(result_df)

    return (f"RMSE (ARIMA): {rmse_arima:.4f}\nRMSE (XGBoost): {rmse_xgb:.4f}\nRMSE (Ensemble): {rmse_ensemble:.4f}",
            result_df.reset_index())

# Inputs
ticker_input = gr.Textbox(label="Ticker", value="AAPL")
arima_weight_input = gr.Slider(label="ARIMA Weight (%)", minimum=0, maximum=100, value=70)
start_input = gr.Textbox(label="Start Date (YYYY-MM-DD)", value="2020-08-01")
end_input = gr.Textbox(label="End Date (YYYY-MM-DD)", value="2025-08-01")

with gr.Blocks() as demo:
    gr.Markdown("## 📈 Stock Price Forecasting (Multimodal ARIMA + XGBoost Ensemble)")

    with gr.Row():
        ticker = ticker_input
        arima_weight = arima_weight_input
        start_date = start_input
        end_date = end_input
        get_feat_button = gr.Button("🔍 Get Feature Groups")

    ohlcv_multiselect = gr.CheckboxGroup([], label="OHLCV", interactive=True)
    ohlcv_der_multiselect = gr.CheckboxGroup([], label="OHLCV Derived", interactive=True)
    gt_multiselect = gr.CheckboxGroup([], label="Google Trends", interactive=True)
    gt_der_multiselect = gr.CheckboxGroup([], label="Google Trends Derived", interactive=True)

    get_feat_button.click(
        fn=get_feature_groups,
        inputs=[ticker, arima_weight, start_date, end_date],
        outputs=[ohlcv_multiselect, ohlcv_der_multiselect, gt_multiselect, gt_der_multiselect]
    )

    with gr.Row():
        forecast_button = gr.Button("📊 Run Forecast")
        output_rmse = gr.Textbox(label="Model RMSEs", interactive=False)
        output_table = gr.Dataframe(label="Forecast Results")

    forecast_button.click(
        fn=run_forecast,
        inputs=[ticker, arima_weight, start_date, end_date,
                ohlcv_multiselect, ohlcv_der_multiselect, gt_multiselect, gt_der_multiselect],
        outputs=[output_rmse, output_table]
    )

demo.launch()
