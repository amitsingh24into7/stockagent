import os
import sqlite3
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from langchain_core.messages import AIMessage, HumanMessage

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Define a state
class State(TypedDict):
    messages: Annotated[list, "Messages between user and agent"]
    sql_query: Annotated[str, "SQL Query derived from question"]
    sql_result: Annotated[str, "Raw result from SQL DB"]
    final_answer: Annotated[str, "Natural language answer"]

# Step 1: Generate SQL from user question
def write_query(state: State) -> State:
    llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="llama3-8b-8192")
    user_question = state["messages"][-1].content
    prompt = f"""You are working with SQLite.

Write ONLY the SQL query (no explanation, no backticks, no markdown formatting).

Use the correct SQLite syntax to answer the question:
{user_question}
"""
    sql_query = llm.invoke(prompt).content.strip()

    # Safety: remove triple backticks if still present
    if sql_query.startswith("```sql") or sql_query.startswith("```"):
        sql_query = sql_query.strip("`").replace("```sql", "").replace("```", "").strip()

    return {**state, "sql_query": sql_query}

# Step 2: Execute SQL
def execute_query(state: State) -> State:
    db = SQLDatabase.from_uri("sqlite:///db/stock_data.db")
    try:
        df = pd.read_sql_query(state["sql_query"], db._engine)
        return {**state, "sql_result": df.to_markdown(index=False)}
    except Exception as e:
        return {**state, "sql_result": f"SQL Error: {e}"}

# Step 3: Generate Natural Language Answer
def generate_answer(state: State) -> State:
    llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="llama3-8b-8192")
    question = state["messages"][-1].content
    sql_query = state["sql_query"]
    result = state["sql_result"]
    if "SQL Error" in result:
        answer = result
    else:
        prompt = f"The user asked: {question}\n\nWe got this SQL result:\n{result}\n\nWrite a clear answer for the user."
        answer = llm.invoke(prompt).content.strip()
    return {**state, "final_answer": answer}

# Create graph
graph_builder = StateGraph(State)
graph_builder.add_node("write_query", write_query)
graph_builder.add_node("execute_query", execute_query)
graph_builder.add_node("generate_answer", generate_answer)
graph_builder.set_entry_point("write_query")
graph_builder.add_edge("write_query", "execute_query")
graph_builder.add_edge("execute_query", "generate_answer")
graph_builder.add_edge("generate_answer", END)
graph = graph_builder.compile()

# Main function
def run_interactive():
    print("📊 Ask me about your stock data (type 'exit' to quit):")
    while True:
        user_input = input("🧑‍💻 You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        state = {
            "messages": [HumanMessage(content=user_input)],
            "sql_query": "",
            "sql_result": "",
            "final_answer": ""
        }
        final = graph.invoke(state)
        print("🤖 Answer:", final["final_answer"])
        print()

if __name__ == "__main__":
    run_interactive()
