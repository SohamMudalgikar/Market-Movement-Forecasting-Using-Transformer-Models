import yfinance as yf
import pandas as pd

def download_stock_data(tickers, period="5d", interval="1m"):
    all_data = {}
    for ticker in tickers:
        print(f"Downloading data for {ticker}")
        data = yf.download(ticker, period=period, interval=interval)
        all_data[ticker] = data
    return all_data

if __name__ == "__main__":
    tickers = ['META', 'AAPL', 'MSFT', 'AMZN', 'GOOG']
    stock_data = download_stock_data(tickers)
    # Save data to CSV files
    for ticker, data in stock_data.items():
        data.to_csv(f"data/{ticker}_data.csv")
