import os
import sqlite3
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
#import logging
#logging.basicConfig(level=logging.DEBUG)

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

def create_and_populate_db():
    os.makedirs("db", exist_ok=True)
    conn = sqlite3.connect("db/stock_data.db")

    for filename in os.listdir("data/day"):
        if filename.endswith(".csv"):
            try:
                df = pd.read_csv(f"data/day/{filename}")
                df.columns = df.columns.str.strip()
                stock_name = "table_" + filename.replace(".csv", "").replace(".", "_")
                df.to_sql(stock_name, conn, if_exists="replace", index=False)
                print(f"✅ Loaded {filename} as table {stock_name}")
            except Exception as e:
                print(f"❌ Failed to load {filename}: {e}")
    
    conn.commit()
    conn.close()

def run_chat_agent():
    db = SQLDatabase.from_uri("sqlite:///db/stock_data.db")
    llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
    toolkit = SQLDatabaseToolkit(db=db,llm=llm)

    agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True
    )

    print("🤖 Agent is ready! Ask your questions...")
    print("📊 Available Tables:", db.get_usable_table_names())

    while True:
        q = input("You: ")
        if q.lower() in ["exit", "quit"]:
            break
        try:
            result = agent.run(q)   # 🔄 This is the key change
            # Handle different response formats
            print(result)

        except Exception as e:
            print("❌ Error:", e)

def list_tables():
    conn = sqlite3.connect("db/stock_data.db")
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("📦 Tables in DB:", [table[0] for table in tables])
    conn.close()

if __name__ == "__main__":
    #create_and_populate_db()
    list_tables()
    run_chat_agent()
