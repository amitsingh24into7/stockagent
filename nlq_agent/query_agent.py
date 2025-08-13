import os
import pandas as pd
from langchain.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatGroq
from dotenv import load_dotenv

load_dotenv()  # Load from .env if available

def get_query_agent(df):
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment")

    llm = ChatGroq(
        temperature=0,
        model_name="mixtral-8x7b-32768",  # You can also try "llama3-70b-8192"
        groq_api_key=groq_api_key
    )

    agent = create_pandas_dataframe_agent(llm, df, verbose=True)
    return agent

def run_query(agent, query):
    return agent.run(query)
