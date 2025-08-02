import sys
import os

# Add 'utils' directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from stock_forecast_pipeline import StockForecastPipeline
