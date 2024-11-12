# Data Visualization

import pandas as pd
import matplotlib.pyplot as plt

def plot_stock_data(data, ticker):
    plt.figure(figsize=(14, 7))
    plt.plot(data['Close'], label='Close Price')
    plt.plot(data['BB_upper'], label='Bollinger Upper Band', linestyle='--')
    plt.plot(data['BB_lower'], label='Bollinger Lower Band', linestyle='--')
    plt.title(f"{ticker} Stock Price with Bollinger Bands")
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig(f"visualizations/{ticker}_bollinger_bands.png")
    plt.close()

def plot_rsi(data, ticker):
    plt.figure(figsize=(14, 3))
    plt.plot(data['RSI'], label='RSI')
    plt.axhline(70, color='red', linestyle='--', label='Overbought')
    plt.axhline(30, color='green', linestyle='--', label='Oversold')
    plt.title(f"{ticker} Relative Strength Index (RSI)")
    plt.xlabel('Date')
    plt.ylabel('RSI')
    plt.legend()
    plt.savefig(f"visualizations/{ticker}_rsi.png")
    plt.close()

if __name__ == "__main__":
    tickers = ['META', 'AAPL', 'MSFT', 'AMZN', 'GOOG']
    for ticker in tickers:
        print(f"Processing data for {ticker}")
        data = pd.read_csv(f"data/{ticker}_data.csv", parse_dates=['Datetime'])
        data.set_index('Datetime', inplace=True)

        # Ensure columns are numeric
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        for col in numeric_columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        # Drop rows with missing values in numeric columns
        data.dropna(subset=numeric_columns, inplace=True)

        # Remove 'Ticker' column if it exists
        if 'Ticker' in data.columns:
            data.drop(columns=['Ticker'], inplace=True)

        # Verify data
        print(f"Data types for {ticker}:\n{data.dtypes}")
        print(f"First few rows for {ticker}:\n{data.head()}")

        # Calculate indicators
        data = calculate_bollinger_bands(data)
        data = calculate_rsi(data)

        # Plot data
        plot_stock_data(data, ticker)
        plot_rsi(data, ticker)
