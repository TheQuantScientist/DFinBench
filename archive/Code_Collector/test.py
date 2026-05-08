import yfinance as yf
import pandas as pd

# Test với 1 ticker
print("=== SINGLE TICKER ===")
df = yf.download("AAPL", start="2021-01-01", end="2021-01-10", progress=False)

print("\n--- RAW DATA ---")
print(type(df))
print(df)

print("\n--- COLUMNS ---")
print(type(df.columns))
print(df.columns)
print("Type:", type(df.columns))

print("\n--- INDEX ---")
print(type(df.index))
print(df.index)
print("Type:", type(df.index))

print("\n--- INDEX NAME ---")
print(type(df.index.name))
print(df.index.name)

# Kiểm tra MultiIndex
print("\n--- IS MULTIINDEX? ---")
print(isinstance(df.columns, pd.MultiIndex))


# ================================
# FIX MULTIINDEX
# ================================
if isinstance(df.columns, pd.MultiIndex):
    print("\nFixing MultiIndex...")
    df.columns = df.columns.droplevel(1)

# Đổi tên cột
df.columns = [col.lower() for col in df.columns]
df.index.name = "timestamp"

print("\n--- AFTER CLEANING ---")
print(df.head())


# ================================
# TEST MULTI-TICKER
# ================================
print("\n\n=== MULTI TICKER ===")
df_multi = yf.download(["AAPL", "MSFT"], start="2021-01-01", end="2021-01-10", progress=False)

print("\n--- RAW MULTI DATA ---")
print(df_multi)

print("\n--- MULTI COLUMNS ---")
print(df_multi.columns)

print("\n--- ACCESS EXAMPLE ---")
print("Close AAPL:")
print(type(df_multi["Close"]["AAPL"].head()))
print(df_multi["Close"]["AAPL"].head())

print("\nClose MSFT:")
print(df_multi["Close"]["MSFT"].head())