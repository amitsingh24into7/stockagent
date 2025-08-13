import pandas as pd
from analysis.fetcher import analyze_all_stocks
from analysis.summarizer import get_trend_summary
from analysis.queries import query_all_by_trend, query_stock_by_trend

def load_symbols(filepath):
    df = pd.read_excel(filepath, sheet_name='Sheet1')
    return df['FinalSymbol'].dropna().tolist()

def main():
    symbol_file = 'data/ind_nifty500list.xlsx'
    symbols = load_symbols(symbol_file)

    data = analyze_all_stocks(symbols)
    data.to_csv("output/full_stock_data.csv", index=False)

    month_summary = get_trend_summary(data, 'Month')
    week_summary = get_trend_summary(data, 'Week')
    day_summary = get_trend_summary(data, 'Day')

    month_summary.to_csv("output/monthly_trend.csv", index=False)
    week_summary.to_csv("output/weekly_trend.csv", index=False)
    day_summary.to_csv("output/daily_trend.csv", index=False)

    print("\n✅ Stocks that went UP in 2023-08:")
    print(query_all_by_trend(month_summary, '2023-08', 'Month', 'up'))

    print("\n📊 RELIANCE.NS trend in Week 2023-30:")
    print(query_stock_by_trend(week_summary, 'RELIANCE.NS', '2023-30', 'Week'))

    print("\n📊 TCS.NS on 2023-07-24:")
    print(query_stock_by_trend(day_summary, 'TCS.NS', '2023-07-24', 'Day'))

if __name__ == "__main__":
    main()
