# stock_ai.py
import streamlit as st
import sqlite3
import pandas as pd
import os
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# -------------------------------
# 🔐 IMPORT DB CONNECTION (Supabase for Auth)
# -------------------------------
try:
    from db.db_connection import get_db_connection, verify_password
except Exception as e:
    st.error("❌ Could not load database connection.")
    st.code(str(e))
    st.stop()

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# -------------------------------
# 🔐 LOGIN CHECK
# -------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🔐 Stock.AI - Login")
    st.markdown("Please log in to access the financial analytics AI.")

    username = st.text_input("📧 Email Address")
    password = st.text_input("🔑 Password", type="password")
    login_btn = st.button("🔓 Login")

    if login_btn:
        if not username or not password:
            st.error("⚠️ Please fill in both fields.")
        else:
            try:
                APP_SLUG = os.getenv("APP_SLUG", "stock-ai")  # Set in .env
                with get_db_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("""
                            SELECT password_hash, valid_until, is_active
                            FROM users
                            WHERE username = %s AND app_slug = %s
                        """, (username, APP_SLUG))
                        row = cur.fetchone()

                if not row:
                    st.error("❌ User not found for this application.")
                else:
                    password_hash, valid_until, is_active = row
                    from datetime import date

                    if not is_active:
                        st.error("❌ Account is inactive.")
                    elif valid_until < date.today():
                        st.error(f"📅 Your access expired on {valid_until}. Contact admin.")
                    elif verify_password(password, password_hash):
                        st.session_state.logged_in = True
                        st.session_state.user_name = username
                        st.session_state.current_page = "stock_ai"
                        st.rerun()
                    else:
                        st.error("❌ Incorrect password.")
            except Exception as e:
                st.error("⚠️ System error. Please try again.")
                st.exception(e)  # Remove in production
    st.stop()  # Don't run the rest of the app

# ------------------ Secure API Key from DB ------------------
def get_api_key_from_db(conn, service="groq"):
    try:
        result = conn.execute("SELECT api_key FROM api_keys WHERE service = ?", (service,)).fetchone()
        if result:
            return result[0]
        else:
            raise ValueError(f"{service.upper()} API key not found in database.")
    except Exception as e:
        st.error(f"🔐 Database error loading API key: {e}")
        return None


# Connect to SQLite database (for stock data only)
import psycopg2
# --- PostgreSQL connection ---

# Load Groq API key from database (not .env)
#groq_api_key = get_api_key_from_db(conn, "groq")
#groq_api_key = "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"
try:
    conn = psycopg2.connect(
        host="192.168.0.110",          # or your DB host (e.g. "db-service" in Kubernetes)
        port="5432",               # default Postgres port
        dbname="stockdb",          # your database name
        user="dbuser",           # your DB username
        password="dbuser@123"    # your DB password
    )
    cursor = conn.cursor()
except Exception as e:
    st.error(f"❌ Failed to connect to PostgreSQL database: {e}")
    st.stop()

# Load Groq API key from database (not .env)
groq_api_key = get_api_key_from_db(conn, "groq")
st.success(groq_api_key)

if not groq_api_key:
    st.error("❌ No Groq API key found in database. Please add it to the `api_keys` table.")
    st.stop()

# LLM setup
try:
    llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")
except Exception as e:
    st.error(f"❌ Failed to initialize LLM: {e}")
    st.stop()

# ------------------ Few-shot Examples ------------------
examples = [
    {
        "question": "What is the average closing price of TCS in July 2025?",
        "sql": """SELECT AVG(Close) AS avg_close
FROM stock_data
WHERE Stock = 'TCS.NS'
  AND substr(Date, 1, 7) = '2025-07';"""
    },
    {
        "question": "What is the Open price of TCS on 31st July August 2025?",
        "sql": """SELECT Open
FROM stock_data
WHERE Stock = 'TCS.NS'
  AND substr(Date, 1, 10) = '2025-07-31';"""
    },
    {
        "question": "Show all data for TCS on 31st July 2025",
        "sql": """SELECT *
FROM stock_data
WHERE Stock = 'TCS.NS'
  AND substr(Date, 1, 10) = '2025-07-31';"""
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
    ORDER BY month
)
SELECT * FROM trend_summary;"""
    }
]

# Dynamically construct few-shot prompt
prompt_examples = "\n\n---\n\n".join(
    f"Question: {ex['question']}\nSQL:\n{ex['sql']}" for ex in examples
)

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

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="📈 Stock AI", layout="wide")
st.title("📈 Stock.AI")

# Add Logout to Sidebar
with st.sidebar:
    st.subheader("Helpers")
    if "selected_example" not in st.session_state:
        st.session_state.selected_example = ""

    for i, ex in enumerate(examples):
        if st.button(f"➡️ {ex['question']}", key=f"example_{i}"):
            st.session_state.selected_example = ex["question"]

    st.markdown("---")
    st.markdown("👆 Click an example to autofill")

    # 🔒 Logout
    st.markdown("---")
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# Input
user_input = st.text_input("🧠 Ask a Query:", value=st.session_state.get("selected_example", ""))

if user_input:
    with st.spinner("💬 Generating SQL..."):
        try:
            result = chain.invoke({"question": user_input})
            sql_query = result.content.strip()
        except Exception as e:
            st.error(f"❌ Failed to generate SQL: {e}")
            st.stop()

    # Validate SQL
    if not sql_query.lower().startswith("select") and "with" not in sql_query.lower():
        st.error("❌ Only SELECT queries are allowed.")
    else:
        try:
            df = pd.read_sql_query(sql_query, conn)
            if df.empty:
                st.warning("⚠️ No results found.")
            else:
                st.success("✅ Data retrieved successfully!")
                st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.error(f"❌ SQL Execution Error:\n\n{e}")