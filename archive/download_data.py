import yfinance as yf
import pandas as pd
import os

# 40 Tickers
STOCKS = {
    'Tech': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSM', 'AVGO', 'ORCL', 'CSCO'],
    'Healthcare': ['JNJ', 'UNH', 'LLY', 'NVO', 'MRK', 'ABBV', 'TMO', 'DHR', 'PFE', 'AZN'],
    'Finance': ['JPM', 'BAC', 'V', 'MA', 'WFC', 'MS', 'GS', 'C', 'AXP', 'SCHW'],
    'Food_Beverage': ['KO', 'PEP', 'MCD', 'SBUX', 'KDP', 'MDLZ', 'HSY', 'GIS', 'NSRGY', 'TSN']
}

START_DATE = '2021-01-01'
END_DATE = '2026-04-24' # Cover all 
SAVE_DIR = './dataset/stock'

os.makedirs(SAVE_DIR, exist_ok=True)

print("Downloading...")
for sector, tickers in STOCKS.items():
    for ticker in tickers:
        print(f"Downloading {ticker} ({sector})...")
        try:
            # API of YFinance, df: pandas DataFrame
            df = yf.download(ticker, start=START_DATE, end=END_DATE, interval='1d', progress=False)
            
            if df.empty:
                print(f"No data for {ticker}")
                continue
            
            # Tackling MultiIndex
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)

            # Naming
            df.index.name = 'timestamp'
            df.columns = [col.lower() for col in df.columns]
            
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            file_path = os.path.join(SAVE_DIR, f"{ticker}_1d_full.csv")
            df.to_csv(file_path)
        except Exception as e:
            print(f" Error occurs while downloading {ticker}: {e}")
