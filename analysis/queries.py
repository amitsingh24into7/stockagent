# analysis/queries.py
def query_all_by_trend(trend_df, period, granularity, trend_type='up'):
    return trend_df[
        (trend_df[granularity] == period) & 
        (trend_df.get(trend_type, 0) > 0)
    ][['Symbol', trend_type]]

def query_stock_by_trend(trend_df, symbol, period, granularity):
    return trend_df[
        (trend_df[granularity] == period) &
        (trend_df['Symbol'] == symbol)
    ]
