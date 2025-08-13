import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_groq import ChatGroq

# Load environment variable
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# 🔧 Function 1: List all stock files
def list_all_stocks(input_str: str) -> str:
    timeframe = input_str.strip().lower()
    path = f"ohlc/{timeframe}"
    if not os.path.exists(path):
        return f"No data available for timeframe: {timeframe}"
    
    files = [f.replace(".csv", "") for f in os.listdir(path) if f.endswith(".csv")]
    if not files:
        return "No stock files found."
    
    return f"Available stocks in '{timeframe}':\n" + "\n".join(files)

# 🔧 Function 2: Stock Summary
def get_stock_summary(input_str: str) -> str:
    timeframe = input_str.strip().lower()
    path = f"ohlc/{timeframe}"
    if not os.path.exists(path):
        return f"No data available for timeframe: {timeframe}"
    
    files = os.listdir(path)
    summary = []
    for file in files:
        df = pd.read_csv(os.path.join(path, file))
        if 'Close' in df.columns:
            summary.append({
                'Stock': file.replace('.csv', ''),
                'Min': df['Close'].min(),
                'Max': df['Close'].max(),
                'Mean': df['Close'].mean()
            })
    if not summary:
        return "No valid data found in CSV files."
    return pd.DataFrame(summary).to_string(index=False)

# 🔧 Function 3: Up/Down analysis
def get_up_down_counts(input_str: str) -> str:
    try:
        parts = input_str.split(",")
        stock_symbol = parts[0].strip()
        timeframe = parts[1].strip() if len(parts) > 1 else "week"
    except:
        return "Input format must be: STOCK_SYMBOL, TIMEFRAME"
    
    filepath = f"ohlc/{timeframe}/{stock_symbol}.csv"
    if not os.path.exists(filepath):
        return f"No data found for stock '{stock_symbol}' in '{timeframe}'"

    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.strftime('%b')  # 'Jan', 'Feb', etc.
    df['Up'] = df['Close'] > df['Open']

    summary = df.groupby('Month')['Up'].agg(['sum', 'count'])
    summary['Down'] = summary['count'] - summary['sum']
    return summary[['sum', 'Down']].rename(columns={'sum': 'Up'}).to_string()

# 🚀 Define tools
tools = [
    Tool(
        name="ListAllStocks",
        func=list_all_stocks,
        description="Input: timeframe (e.g., 'daily', 'weekly', 'monthly'). Lists available stock files for the given frequency."
    ),
    Tool(
        name="GetStockSummary",
        func=get_stock_summary,
        description="Input: timeframe (e.g., 'daily', 'weekly', 'monthly'). Returns min, max, mean of Close for all stocks."
    ),
    Tool(
        name="GetUpDownCounts",
        func=get_up_down_counts,
        description="Input: stock symbol and timeframe separated by comma. Example: 'RELIANCE, weekly'. Returns up/down month counts."
    ),
]

# 🤖 Set up LLM agent
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama3-8b-8192")
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# 🌐 Streamlit UI
st.set_page_config(page_title="Stock Trend AI", layout="wide")
st.title("📈 Stock Trend Analyzer using Groq + LangChain")

query = st.text_input("Ask your query (e.g., 'List all weekly stocks', 'Summary for monthly data', 'Up/Down for TCS, monthly')")

if st.button("Run Query") and query:
    with st.spinner("Processing..."):
        try:
            result = agent.run(query)
            st.text_area("📊 Result", result, height=400)
        except Exception as e:
            st.error(f"❌ Error: {e}")
