# analysis/fatcher.py

import os
import yfinance as yf
import pandas as pd

def fetch_stock_data(symbol, period='10y'):
    df = yf.download(symbol, period=period)
    #stock = yf.Ticker(symbol)
    #df = stock.history(period=period)
    #return hist
    '''
    if df.empty:
        return None
    df['Symbol'] = symbol
    df['Day'] = df.index.to_series().dt.strftime('%Y-%m-%d')
    df['Week'] = df.index.to_series().dt.strftime('%Y-%U')
    df['Month'] = df.index.to_series().dt.strftime('%Y-%m')
    df['Change'] = df['Close'].diff()
    df['Trend'] = df['Change'].apply(lambda x: 'up' if x > 0 else 'down' if x < 0 else 'no change')
    '''
    return df

def analyze_all_stocks(symbols):
    all_data = []
    os.makedirs("output/stocks", exist_ok=True)  # Ensure the folder exists

    for symbol in symbols:
        print(f"Fetching {symbol}")
        try:
            df = fetch_stock_data(symbol)
            if df is not None and not df.empty:
                # Save individual stock data as CSV
                output_path = f"output/stocks/{symbol}.csv"
                df.to_csv(output_path, index=True)
                print(f"Saved {symbol} data to {output_path}")
                
                all_data.append(df)
        except Exception as e:
            print(f"Error with {symbol}: {e}")
    
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
