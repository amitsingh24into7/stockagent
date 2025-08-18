import os
import yfinance as yf
import pandas as pd

# Config
BASE_DIR = "data"
EXCEL_FILE = "data/ind_nifty500list.xlsx"  # Your Excel file
SHEET_NAME = "Sheet1"  # Update if needed
COLUMN_NAME = "FinalSymbol"  # Column name in Excel file
TIMEFRAMES = {
    "day": "1d"
}

# Ensure folders exist
def ensure_directories():
    for tf in TIMEFRAMES:
        os.makedirs(os.path.join(BASE_DIR, tf), exist_ok=True)

# Read stock list from Excel
def get_stock_list_from_excel():
    df = pd.read_excel(EXCEL_FILE, sheet_name=SHEET_NAME)
    stock_list = df[COLUMN_NAME].dropna().unique().tolist()
    return [s.upper() for s in stock_list]

# Fetch and save OHLC data
def fetch_ohlc(stock_list, period="10y"):
    ensure_directories()

    for stock in stock_list:
        print(f"Fetching data for: {stock}")
        try:
            ticker = yf.Ticker(stock)
            for tf, interval in TIMEFRAMES.items():
                df = ticker.history(period=period, interval=interval)
                df.reset_index(inplace=True)

                file_path = os.path.join(BASE_DIR, tf, f"{stock}.csv")
                df.to_csv(file_path, index=False)
                print(f"Saved {tf} data to {file_path}")

        except Exception as e:
            print(f"Error fetching {tf} data for {stock}: {e}")

# Run
if __name__ == "__main__":
    stocks = get_stock_list_from_excel()
    fetch_ohlc(stocks)
