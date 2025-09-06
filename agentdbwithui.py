import os
import sqlite3
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase

# Load API key from .env
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Caching agent setup
@st.cache_resource
def setup_agent():
    db = SQLDatabase.from_uri("sqlite:///db/stock_data.db")
    llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    agent = create_sql_agent(llm=llm, toolkit=toolkit, verbose=True)
    return agent, db, llm

# Page config
st.set_page_config(page_title="Stock AI Chat", layout="wide")
st.title("📊 Chat with Your Stock Market Data")

# Setup
agent, db, llm = setup_agent()

# Sidebar - Chat history style placeholder
with st.sidebar:
    st.header("📁 Chat Context / Info")
    st.markdown("Available Tables in DB:")
    tables = db.get_usable_table_names()
    for t in tables:
        st.markdown(f"- `{t}`")
    st.markdown("---")
    st.caption("Ask questions like:")
    st.code("Show monthly average price", language="sql")
    st.code("Top 5 highest volume days", language="sql")

# Main input
user_question = st.text_input("💬 Ask your question about stock data:")

if user_question:
    try:
        with st.spinner("🤖 Thinking..."):
            # Step 1: Agent responds
            response = agent.run(user_question)

            # Step 2: Try to get the SQL query only
            sql_prompt = f"Write only the SQL query (no explanation) that would answer: {user_question}"
            sql_query = llm.invoke(sql_prompt).content.strip()
            print(sql_query)

        # Show LLM response
        st.success("✅ Answer:")
        #st.markdown(response)

        #st.success("✅ Answer:")

        try:
            # Try parsing the response into a pandas DataFrame using LangChain's tool result
            # This assumes that the final result is already a dict-like object or table (depends on the tool)
            # You can attempt to convert it directly to a DataFrame if it's a list of dicts or tuples

            if isinstance(response, list):
                df = pd.DataFrame(response)
                st.dataframe(df, use_container_width=True)
                
                # Allow CSV download
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download as CSV",
                    data=csv,
                    file_name="result.csv",
                    mime="text/csv"
                )
            else:
                # If response is just plain text (e.g., a summary or explanation), show as-is
                st.markdown(response)

        except Exception as e:
            # Fallback: just show raw response
            st.markdown(response)
            st.error(f"⚠️ Could not format response as table: {e}")
    except Exception as e:
        st.error(f"⚠️ Could not format response as table: {e}")
