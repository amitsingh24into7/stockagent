import streamlit as st
import sqlite3
import pandas as pd
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Connect to SQLite database
conn = sqlite3.connect("db/stock_data.db")

# LLM setup
llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="llama3-8b-8192")

# Few-shot examples (include various prompt phrasings)
examples = [
    {
        "question": "What is the average closing price of TCS in August 2015?",
        "sql": """SELECT AVG(Close) AS avg_close
FROM stock_data
WHERE Stock = 'TCS.NS'
  AND substr(Date, 1, 7) = '2015-08';"""
    },
    {
        "question": "What is the Open price of TCS on 3rd August 2025?",
        "sql": """SELECT Open
FROM stock_data
WHERE Stock = 'TCS.NS'
  AND substr(Date, 1, 10) = '2025-08-03';"""
    },
    {
        "question": "Show all data for TCS on 3rd August 2015",
        "sql": """SELECT *
FROM stock_data
WHERE Stock = 'TCS.NS'
  AND substr(Date, 1, 10) = '2015-08-03';"""
    },
    {
        "question": "Show monthly average closing price of HDFCBANK",
        "sql": """SELECT substr(Date, 1, 7) AS month, AVG(Close) AS avg_close
FROM stock_data
WHERE Stock = 'HDFCBANK.NS'
GROUP BY month
ORDER BY month;"""
    },
    {
        "question": "Calculate month-wise percentage change in average closing price for HDFCBANK",
        "sql": """WITH monthly_avg AS (
    SELECT 
        substr(Date, 1, 4) AS year,
        substr(Date, 6, 2) AS month,
        Stock AS stock_symbol,
        AVG(Close) AS avg_close
    FROM stock_data
    WHERE Stock = 'HDFCBANK.NS'
    GROUP BY year, month, stock_symbol
),
monthly_change AS (
    SELECT 
        year,
        month,
        stock_symbol,
        avg_close,
        LAG(avg_close) OVER (PARTITION BY stock_symbol ORDER BY year, month) AS prev_avg_close
    FROM monthly_avg
)
SELECT 
    month,
    ROUND(AVG((avg_close - prev_avg_close) / prev_avg_close * 100), 2) AS avg_pct_change
FROM monthly_change
WHERE prev_avg_close IS NOT NULL
GROUP BY month
ORDER BY month;"""
    },
    {
        "question": "Calculate month-wise percentage change in average closing price for HDFCBANK in 2024",
        "sql": """WITH monthly_avg AS (
    SELECT 
        substr(Date, 1, 4) AS year,
        substr(Date, 6, 2) AS month,
        Stock AS stock_symbol,
        AVG(Close) AS avg_close
    FROM stock_data
    WHERE Stock = 'HDFCBANK.NS'
      AND substr(Date, 1, 4) = '2024'
    GROUP BY year, month, stock_symbol
),
monthly_change AS (
    SELECT 
        year,
        month,
        stock_symbol,
        avg_close,
        LAG(avg_close) OVER (PARTITION BY stock_symbol ORDER BY year, month) AS prev_avg_close
    FROM monthly_avg
)
SELECT 
    month,
    ROUND(AVG((avg_close - prev_avg_close) / prev_avg_close * 100), 2) AS avg_pct_change
FROM monthly_change
WHERE prev_avg_close IS NOT NULL
GROUP BY month
ORDER BY month;"""
    },
    {
        "question": "Show month-wise up/down trend summary for HDFCBANK over 10 years",
        "sql": """WITH monthly_avg AS (
    SELECT 
        substr(Date, 1, 4) AS year,
        substr(Date, 6, 2) AS month,
        Stock AS stock_symbol,
        AVG(Close) AS avg_close
    FROM stock_data
    WHERE Stock = 'HDFCBANK.NS'
    GROUP BY year, month, stock_symbol
),
monthly_change AS (
    SELECT 
        year,
        month,
        stock_symbol,
        avg_close,
        LAG(avg_close) OVER (PARTITION BY stock_symbol ORDER BY year, month) AS prev_avg_close
    FROM monthly_avg
),
monthly_trend AS (
    SELECT 
        year,
        month,
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
        CASE month
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
    GROUP BY month, stock_symbol
    ORDER BY month;
)
SELECT * FROM trend_summary;"""
    }
]


# Dynamically construct few-shot prompt
prompt_examples = "\n\n---\n\n".join(
    f"Question: {ex['question']}\nSQL:\n{ex['sql']}" for ex in examples
)

# LangChain prompt template

prompt_template = f"""
You are an expert SQLite SQL generator for stock market time-series data.

Use only valid SQLite syntax — no markdown, no explanation, just SQL.

**Important Notes:**
- Table name: `stock_data`
- Fields: Date, Stock, Open, High, Low, Close, Volume, Dividends, Stock_Splits
- ⚠️ Date is in format: 'YYYY-MM-DD HH:MM:SS+05:30' (e.g., 2015-07-31 00:00:00+05:30)
- ✅ When filtering on a specific date, use: `substr(Date, 1, 10) = 'YYYY-MM-DD'`
- ✅ For month filters, use: `substr(Date, 1, 7)` or `strftime('%Y-%m', Date)`

{prompt_examples}

---

Question: {{question}}
SQL:
"""


prompt = PromptTemplate(input_variables=["question"], template=prompt_template)
chain = prompt | llm

# ------------------ Streamlit Layout ------------------
st.set_page_config(page_title="📈 Stock AI", layout="wide")
st.title("TradeShala AI")

# Sidebar: few-shot prompt explorer
with st.sidebar:
    st.subheader("Examples")
    if "selected_example" not in st.session_state:
        st.session_state.selected_example = ""

    for i, ex in enumerate(examples):
        if st.button(f"➡️ {ex['question']}", key=f"example_{i}"):
            st.session_state.selected_example = ex["question"]

    st.markdown("---")
    st.markdown("👆 Click an example to autofill")

# Text input box
user_input = st.text_input("🧠 Ask a Your Query:", value=st.session_state.get("selected_example", ""))

if user_input:
    with st.spinner("💬 Generating SQL from your question..."):
        result = chain.invoke({"question": user_input})

    sql_query = result.content.strip()
    
    # Display SQL
    with st.expander("🧪 Preview Generated SQL"):
        st.code(sql_query, language="sql")

    # Safeguard before running
    if not sql_query.lower().strip().startswith("select") and "with" not in sql_query.lower():
        st.error("❌ Generated SQL does not seem valid. Please refine your question.")
    else:
        try:
            df = pd.read_sql_query(sql_query, conn)
            if df.empty:
                st.warning("⚠️ No results found.")
            else:
                st.success("✅ Query executed successfully!")
                st.dataframe(df, use_container_width=True)

                # Optional: download result
                # csv = df.to_csv(index=False).encode("utf-8")
                # st.download_button("📥 Download CSV", data=csv, file_name="result.csv", mime="text/csv")
        except Exception as e:
            st.error(f"❌ SQL Execution Error:\n\n{e}")
