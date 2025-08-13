import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain.agents import initialize_agent, Tool
from langchain_groq import ChatGroq
from langchain.agents.agent_types import AgentType
from datetime import datetime

# Load API Key
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("GROQ_API_KEY not found. Please set it in .env file.")
    st.stop()

# Streamlit UI
st.title("📈 Stock Trend Analyzer (LLM + Tools)")
frequency = st.selectbox("Select Timeframe", ["day", "week", "month"])
user_prompt = st.text_input("Ask a question (e.g., 'Show month-wise stock performance')")

# === Core Functions ===
def list_all_stocks(frequency: str) -> pd.DataFrame:
    folder = os.path.join("data", frequency)
    df_list = []
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            path = os.path.join(folder, file)
            df = pd.read_csv(path, parse_dates=["Date"])
            df["Stock"] = file.replace(".csv", "")
            df_list.append(df)
    return pd.concat(df_list, ignore_index=True)

def get_up_down_counts(df: pd.DataFrame) -> pd.DataFrame:
    df["Month"] = df["Date"].dt.strftime("%B")
    df["Up"] = df["Close"] > df["Open"]
    summary = df.groupby("Month")["Up"].agg(
        UpCount=lambda x: (x == True).sum(),
        DownCount=lambda x: (x == False).sum()
    ).reset_index()
    return summary

def get_stock_summary(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("Stock")["Close"].agg(["mean", "min", "max"]).reset_index()

# === Tool Definitions ===
tools = [
    Tool.from_function(
        func=lambda: list_all_stocks(frequency),
        name="ListAllStocks",
        description="Load all stock data from the selected frequency folder (day/week/month)."
    ),
    Tool.from_function(
        func=lambda: get_up_down_counts(list_all_stocks(frequency)),
        name="GetUpDownCounts",
        description="Get monthly Up and Down counts using OHLC data."
    ),
    Tool.from_function(
        func=lambda: get_stock_summary(list_all_stocks(frequency)),
        name="GetStockSummary",
        description="Get mean/min/max of closing prices for each stock."
    )
]

# === Initialize Agent ===
llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="llama3-8b-8192")
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# === Run LLM Agent ===
if user_prompt:
    with st.spinner("Analyzing..."):
        try:
            response = agent.run(user_prompt)
            st.write("### 🔍 Response:")
            st.write(response)
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
