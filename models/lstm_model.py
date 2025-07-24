import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

def prepare_lstm_data(df, features, target='Close', window_size=30):
    data = df[features + [target]].copy()
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(window_size, len(data_scaled)):
        X.append(data_scaled[i - window_size:i, :-1])
        y.append(data_scaled[i, -1])
    X, y = np.array(X), np.array(y)
    return X, y, scaler

def train_lstm_model(X_train, y_train, epochs=30, batch_size=16):
    model = Sequential()
    model.add(LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=32))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')

    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[early_stop])

    return model

def predict_lstm(model, df, features, scaler, window_size=30):
    data = df[features + ['Close']].copy()
    data_scaled = scaler.transform(data)
    X_pred = []
    for i in range(window_size, len(data_scaled)):
        X_pred.append(data_scaled[i - window_size:i, :-1])
    X_pred = np.array(X_pred)

    pred_scaled = model.predict(X_pred)
    pred_full = np.zeros((len(pred_scaled), data_scaled.shape[1]))
    pred_full[:, -1] = pred_scaled.flatten()
    preds = scaler.inverse_transform(pred_full)[:, -1]

    return preds
