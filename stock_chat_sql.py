import sqlite3
import pandas as pd
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Connect to SQLite DB
conn = sqlite3.connect("db/stock_data.db")

# Set up LLM
llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="llama3-8b-8192")

# Prompt Template
prompt_template = """
You are an expert SQLite SQL generator for stock market time-series data.

Only return the valid SQLite SQL query. Do not include any explanation, markdown, or comments.

Your data table is `stock_data` with these columns:
- Date (format: 2015-07-31 00:00:00+05:30)
- Stock (e.g., HDFCBANK.NS, TCS.NS)
- Open, High, Low, Close, Volume, Dividends, Stock_Splits
- Optional: RSI, MA5, MA20 (may be added later)

Use correct SQLite syntax only.

Examples:

---

Question: What is the average closing price of TCS in August 2015?
SQL:
SELECT AVG(Close) AS avg_close FROM stock_data WHERE Stock = 'TCS.NS' AND Date >= '2015-08-01' AND Date < '2015-09-01';

---

Question: Show monthly average closing price of HDFCBANK
SQL:
SELECT substr(Date, 1, 7) AS month, AVG(Close) AS avg_close FROM stock_data WHERE Stock = 'HDFCBANK.NS' GROUP BY month ORDER BY month;

---

Question: Which months consistently gave positive percentage change in average closing price over the last 10 years?
SQL:
WITH monthly_avg AS (
    SELECT
        strftime('%Y', Date) AS year,
        strftime('%m', Date) AS month,
        AVG(Close) AS avg_close
    FROM stock_data
    WHERE Stock = 'HDFCBANK.NS'
      AND Date >= '2012-01-01'
    GROUP BY year, month
),
monthly_pct_change AS (
    SELECT
        year,
        month,
        avg_close,
        ROUND(
            (avg_close - LAG(avg_close) OVER (PARTITION BY month ORDER BY year)) /
            LAG(avg_close) OVER (PARTITION BY month ORDER BY year) * 100,
            2
        ) AS pct_change
    FROM monthly_avg
),
positive_months AS (
    SELECT year, month
    FROM monthly_pct_change
    WHERE pct_change > 0
)
SELECT month, COUNT(*) AS times_positive
FROM positive_months
GROUP BY month
ORDER BY times_positive DESC;

---

Question: Show month-wise up/down trend summary over 10 years for AARTIIND
SQL:
WITH monthly_avg AS (
    SELECT
        strftime('%Y', Date) AS year,
        strftime('%m', Date) AS month_num,
        strftime('%m', Date) || '-' || Stock AS key,
        Stock AS stock_symbol,
        AVG(Close) AS avg_close
    FROM stock_data
    WHERE Stock = 'AARTIIND.NS'
    GROUP BY year, month_num, stock_symbol
),
monthly_change AS (
    SELECT
        year,
        month_num,
        stock_symbol,
        avg_close,
        LAG(avg_close) OVER (PARTITION BY stock_symbol ORDER BY year, month_num) AS prev_avg_close
    FROM monthly_avg
),
monthly_trend AS (
    SELECT
        year,
        month_num,
        stock_symbol,
        CASE
            WHEN avg_close > prev_avg_close THEN 'up'
            WHEN avg_close < prev_avg_close THEN 'down'
            ELSE 'neutral'
        END AS trend
    FROM monthly_change
    WHERE prev_avg_close IS NOT NULL
),
trend_summary AS (
    SELECT
        CASE month_num
            WHEN '01' THEN 'January'
            WHEN '02' THEN 'February'
            WHEN '03' THEN 'March'
            WHEN '04' THEN 'April'
            WHEN '05' THEN 'May'
            WHEN '06' THEN 'June'
            WHEN '07' THEN 'July'
            WHEN '08' THEN 'August'
            WHEN '09' THEN 'September'
            WHEN '10' THEN 'October'
            WHEN '11' THEN 'November'
            WHEN '12' THEN 'December'
        END AS month,
        stock_symbol,
        COUNT(DISTINCT year) AS total_years,
        SUM(CASE WHEN trend = 'up' THEN 1 ELSE 0 END) AS up_trends,
        SUM(CASE WHEN trend = 'down' THEN 1 ELSE 0 END) AS down_trends,
        CASE
            WHEN SUM(CASE WHEN trend = 'up' THEN 1 ELSE 0 END) > SUM(CASE WHEN trend = 'down' THEN 1 ELSE 0 END) THEN 'up'
            WHEN SUM(CASE WHEN trend = 'down' THEN 1 ELSE 0 END) > SUM(CASE WHEN trend = 'up' THEN 1 ELSE 0 END) THEN 'down'
            ELSE 'neutral'
        END AS predominant_trend
    FROM monthly_trend
    GROUP BY month_num, stock_symbol
    ORDER BY month_num
)
SELECT * FROM trend_summary;

Question: {question}
SQL:
"""

prompt = PromptTemplate(input_variables=["question"], template=prompt_template)
chain = prompt | llm

# CLI Loop
print("📊 Ask me about your stock data (type 'exit' to quit):")
while True:
    user_input = input("🧑‍💻 You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    try:
        # Use new invoke pattern
        response = chain.invoke({"question": user_input})
        sql_query = response.content.strip() if hasattr(response, "content") else str(response).strip()

        print(f"\n📜 SQL:\n{sql_query}\n")

        # Run SQL
        df = pd.read_sql_query(sql_query, conn)

        if df.empty:
            print("⚠️ No results found.")
        else:
            print("📈 Result:")
            print(df.to_markdown(index=False))
            df.to_csv("last_result.csv", index=False)
            print("✅ Saved as 'last_result.csv'")
    except Exception as e:
        print(f"❌ SQL Error: {e}")
