import streamlit as st
import os
import pandas as pd
import difflib
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from langchain_groq import ChatGroq


# Load environment variables
load_dotenv()

# === Streamlit Config ===
st.set_page_config(page_title="📊 Stock OHLC Analyst", layout="wide")
st.title("📊 Stock OHLC Analyst (LLM → Pandas Code Execution)")

# === Constants ===
DATA_DIR = "data"

# === Utility Functions ===
def match_column(word, actual_columns):
    matches = difflib.get_close_matches(word.lower(), [col.lower() for col in actual_columns], n=1, cutoff=0.6)
    if matches:
        for col in actual_columns:
            if col.lower() == matches[0]:
                return col
    return None

def suggest_column_insights(df):
    suggestions = []
    for col in df.columns:
        dtype = df[col].dtype
        if pd.api.types.is_numeric_dtype(dtype):
            suggestions.append(f"📈 Show average or sum of `{col}`")
        elif pd.api.types.is_categorical_dtype(dtype) or df[col].nunique() <= 10:
            suggestions.append(f"📊 Count by category in `{col}`")
        elif "date" in col.lower() or "year" in col.lower():
            suggestions.append(f"📆 Trend over `{col}`")
    return suggestions

def describe_column(df, col):
    if pd.api.types.is_numeric_dtype(df[col]):
        return f"Numerical, mean: {df[col].mean():.2f}"
    else:
        top_vals = df[col].value_counts().nlargest(3).index.tolist()
        return f"Categorical, top: {', '.join(map(str, top_vals))}"

def get_stock_list(freq):
    folder = os.path.join(DATA_DIR, freq)
    if not os.path.exists(folder):
        return []
    return [f.replace(".csv", "") for f in os.listdir(folder) if f.endswith(".csv")]

def load_stock_data(stock, freq):
    path = os.path.join(DATA_DIR, freq, f"{stock}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        return df
    return None

from datetime import datetime

def run_agent(df, user_query):
    # Custom fallback check
    if "month-wise performance" in user_query.lower() or "went up" in user_query.lower():
        try:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df[df['Date'].dt.year >= datetime.now().year - 10].copy()
            df['Year'] = df['Date'].dt.year
            df['Month'] = df['Date'].dt.month
            df['Performance'] = df.apply(
                lambda row: 'Up' if row['Close'] > row['Open'] else ('Down' if row['Close'] < row['Open'] else 'Same'),
                axis=1
            )
            summary = df.groupby(['Year', 'Month', 'Performance']).size().unstack(fill_value=0).reset_index()
            code_used = """
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df[df['Date'].dt.year >= datetime.now().year - 10].copy()
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Performance'] = df.apply(
    lambda row: 'Up' if row['Close'] > row['Open'] else ('Down' if row['Close'] < row['Open'] else 'Same'),
    axis=1
)
result = df.groupby(['Year', 'Month', 'Performance']).size().unstack(fill_value=0).reset_index()
"""
            return summary, code_used
        except Exception as e:
            return f"Error: {e}", ""
    
    # Normal LLM mode
    prompt = f"""
You are a Stock Analyzer AI. A user has provided a DataFrame named `df` containing OHLC stock data.

The first few rows of the data are:
{df.head(5).to_string(index=False)}

Generate a valid, executable one-line or multi-line Pandas expression to answer the user’s question.

Assumptions:
- Date filtering uses `.dt.year`, `.dt.month`, etc.
- For "closing price", use `Close`; "opening price" is `Open`.
- Prefer `.groupby()` for monthly/yearly stats.
- Avoid markdown or explanations. Return only Python code.

User Question:
\"\"\"{user_query}\"\"\"
"""
    llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama3-8b-8192")
    response = llm.invoke(prompt)
    response_text = response.content.strip() if hasattr(response, "content") else str(response).strip()

    try:
        local_env = {"df": df.copy(), "pd": pd, "datetime": datetime, "plt": plt}
        exec(f"result = {response_text}", {}, local_env)
        return local_env["result"], response_text
    except Exception as e:
        return f"Error: {str(e)}", response_text

# === Sidebar Inputs ===
st.sidebar.markdown("## 📁 Select Data")
freq = st.sidebar.selectbox("Select Timeframe", ["day", "week", "month"])
stock_list = get_stock_list(freq)
stock_list_with_all = [""] + ["All"] + stock_list
stock = st.sidebar.selectbox("Select Stock", stock_list_with_all)
load_data = st.sidebar.button("📂 Load Data")

# === Data Loading ===
if 'dfs' not in st.session_state:
    st.session_state.dfs = {}

if load_data and stock:
    st.session_state.dfs = {}
    if stock == "All":
        for stk in stock_list:
            df_temp = load_stock_data(stk, freq)
            if df_temp is not None:
                st.session_state.dfs[stk] = df_temp
            else:
                st.warning(f"⚠️ Could not load data for {stk}")
    else:
        df = load_stock_data(stock, freq)
        if df is not None:
            st.session_state.dfs[stock] = df
        else:
            st.error(f"❌ Failed to load data for {stock}")

# === Main App UI ===
dfs = st.session_state.get("dfs", {})
if dfs:
    #st.write("✅ DataFrame(s) loaded")

    user_query = st.text_area("🔎 Ask about the stock(s)", height=100)

    if st.button("🚀 Submit Query"):
        all_results = []
        last_code = ""
        for stock_name, df in dfs.items():
            #st.markdown(f"---\n### 📊 Analyzing `{stock_name}.csv`")
            result, code = run_agent(df, user_query)

            if isinstance(result, pd.DataFrame) and not result.empty:
                result.insert(0, "Stock", stock_name)
                all_results.append(result)
            elif isinstance(result, str) and result.startswith("Error:"):
                st.error(f"❌ Error for {stock_name}: {result}")

            last_code = code  # This will be shown at the bottom

        if all_results:
            st.markdown("## ✅ Combined Result")
            final_df = pd.concat(all_results, ignore_index=True)
            st.dataframe(final_df)
        else:
            st.warning("❌ No results matched your query across any stocks.")

        if last_code:
            with st.expander("🧠 Generated Pandas Code"):
                st.code(last_code, language="python")

else:
    st.info("🕵️‍♂️ Please select a stock and click 'Load Data' to begin.")
