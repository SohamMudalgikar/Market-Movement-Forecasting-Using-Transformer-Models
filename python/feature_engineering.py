# Feature Engineering

import pandas as pd
import numpy as np

def calculate_bollinger_bands(data, window=14, num_std=2):
    rolling_mean = data['Close'].rolling(window).mean()
    rolling_std = data['Close'].rolling(window).std()
    data['BB_upper'] = rolling_mean + (rolling_std * num_std)
    data['BB_lower'] = rolling_mean - (rolling_std * num_std)
    data['BB_width'] = data['BB_upper'] - data['BB_lower']
    return data

def calculate_rsi(data, window=14):
    delta = data['Close'].diff(1)
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    return data

def calculate_roc(data, window=14):
    data['ROC'] = data['Close'].pct_change(periods=window) * 100
    return data

def preprocess_data(data):
    data = calculate_bollinger_bands(data)
    data = calculate_rsi(data)
    data = calculate_roc(data)
    data.dropna(inplace=True)
    # Normalize features
    features = ['Close', 'BB_width', 'RSI', 'ROC', 'Volume']
    data_mean = data[features].mean()
    data_std = data[features].std()
    data[features] = (data[features] - data_mean) / data_std
    return data, data_mean, data_std

# src/feature_engineering.py

import pandas as pd
import numpy as np

# Existing function definitions...

if __name__ == "__main__":
    tickers = ['META', 'AAPL', 'MSFT', 'AMZN', 'GOOG']
    all_data = []
    stats = {}
    for ticker in tickers:
        print(f"Processing data for {ticker}")
        data = pd.read_csv(f"data/{ticker}_data.csv", parse_dates=['Datetime'])
        print(f"Initial data columns: {data.columns}")

        # Set 'Datetime' as index
        data.set_index('Datetime', inplace=True)

        # Convert numeric columns to proper data types
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        for col in numeric_columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        # Drop rows with missing values
        data.dropna(subset=numeric_columns, inplace=True)

        print(f"Data types after conversion:\n{data.dtypes}")
        print(data.head())

        data, mean, std = preprocess_data(data)
        data['Ticker'] = ticker
        all_data.append(data)
        stats[ticker] = {'mean': mean, 'std': std}

    combined_data = pd.concat(all_data)
    combined_data.to_csv("data/processed_stock_data.csv")
    # Save stats for inverse transformations
    pd.to_pickle(stats, "data/data_stats.pkl")

