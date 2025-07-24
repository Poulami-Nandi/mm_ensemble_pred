import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def train_xgboost(X, y, params=None):
    """
    Train an XGBoost regressor.
    
    Args:
        X: Feature matrix (numpy array or DataFrame)
        y: Target array
        params: Dictionary of hyperparameters
    
    Returns:
        Trained XGBoost model
    """
    if params is None:
        params = {
            "objective": "reg:squarederror",
            "n_estimators": 100,
            "learning_rate": 0.05,
            "max_depth": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42
        }
        
    model = xgb.XGBRegressor(**params)
    model.fit(X, y)
    return model

def predict_xgboost(model, X_test):
    """
    Predict using trained XGBoost model.
    
    Args:
        model: Trained XGBoost model
        X_test: Features for prediction
    
    Returns:
        Predicted values (numpy array)
    """
    return model.predict(X_test)

def train_and_forecast_xgb(X, y, forecast_days=7):
    """
    Split train/test and run XGBoost to predict last N days.
    
    Args:
        X: Feature matrix
        y: Target array
        forecast_days: Number of days to predict from end
    
    Returns:
        Tuple (forecast, actual)
    """
    X_train, X_test = X[:-forecast_days], X[-forecast_days:]
    y_train, y_test = y[:-forecast_days], y[-forecast_days:]
    
    model = train_xgboost(X_train, y_train)
    y_pred = predict_xgboost(model, X_test)
    
    return y_pred, y_test
