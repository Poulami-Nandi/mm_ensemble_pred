import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

def train_arima_model(series, order=(5,1,0)):
    """
    Trains an ARIMA model on the given univariate series.
    Args:
        series: pandas Series of prices (e.g. closing price).
        order: ARIMA order tuple (p, d, q).
    Returns:
        Trained ARIMA model.
    """
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    return model_fit

def predict_arima(model_fit, steps=7):
    """
    Forecasts next 'steps' values using the ARIMA model.
    Args:
        model_fit: fitted ARIMA model.
        steps: number of future steps to forecast.
    Returns:
        Array of forecasted values.
    """
    forecast = model_fit.forecast(steps=steps)
    return forecast.values if hasattr(forecast, 'values') else forecast

def rolling_arima_forecast(series, train_size=0.8, order=(5,1,0), forecast_horizon=7):
    """
    Trains ARIMA on initial portion and forecasts next N steps.
    Args:
        series: time series (e.g. closing prices)
        train_size: float between 0 and 1, percent of data used for training.
        order: ARIMA(p,d,q)
        forecast_horizon: number of future days to predict
    Returns:
        Tuple (forecasted_values, test_values)
    """
    n_train = int(len(series) * train_size)
    train, test = series[:n_train], series[n_train:n_train+forecast_horizon]
    model_fit = train_arima_model(train, order=order)
    forecast = predict_arima(model_fit, steps=forecast_horizon)
    return forecast, test.values
