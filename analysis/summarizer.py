# analysis/summarizer.py

def get_trend_summary(df, granularity='Month'):
    return df.groupby([granularity, 'Symbol', 'Trend']).size().unstack(fill_value=0).reset_index()
