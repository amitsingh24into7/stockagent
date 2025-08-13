# query_ui.py

import streamlit as st
import pandas as pd
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
import os

# Load LLM
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama3-8b-8192")

st.title("📈 Stock Trend Query Assistant")

csv_option = st.selectbox("Select Trend Summary File", [
    "output/monthly_trend.csv",
    "output/weekly_trend.csv",
    "output/daily_trend.csv",
    "output/full_stock_data.csv"
])

user_question = st.text_input("Ask a question about stock trends:")

if csv_option and user_question:
    try:
        df = pd.read_csv(csv_option)
        agent = create_pandas_dataframe_agent(llm, df, verbose=True,allow_dangerous_code=True )

        with st.spinner("Thinking..."):
            answer = agent.run(user_question)
        st.success("Answer:")
        st.write(answer)
    except Exception as e:
        st.error(f"Error: {e}")
