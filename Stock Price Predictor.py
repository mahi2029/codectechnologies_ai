import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Step 1: Download Data
def download_stock_data(ticker, start='2015-01-01', end='2024-01-01'):
    data = yf.download(ticker, start=start, end=end)
    return data['Close'].values.reshape(-1, 1), data

# Step 2: Preprocess Data
def prepare_lstm_data(data, window_size=60):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i - window_size:i, 0])
        y.append(scaled_data[i, 0])
    
    X = np.array(X)
    y = np.array(y)
    
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

# Step 3: Build LSTM Model
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Step 4: Main Function
def main():
    ticker = 'AAPL'
    raw_data, df = download_stock_data(ticker)
    
    X, y, scaler = prepare_lstm_data(raw_data)
    
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    model = build_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=10, batch_size=32)
    
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
    real = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    plt.figure(figsize=(12, 6))
    plt.plot(real, label='Real Price')
    plt.plot(predictions, label='Predicted Price')
    plt.title(f"{ticker} Stock Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
