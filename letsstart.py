import os
import sqlite3
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain.chat_models import init_chat_model

# Load API key from .env
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

db = SQLDatabase.from_uri("sqlite:///db/stock_data.db")
print(db.dialect)
print(db.get_usable_table_names())
#db.run("SELECT * FROM Artist LIMIT 10;")
#llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="llama3-8b-8192")
#toolkit = SQLDatabaseToolkit(db=db, llm=llm)
#agent = create_sql_agent(llm=llm, toolkit=toolkit, verbose=True)

from typing_extensions import TypedDict


class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str

llm = init_chat_model("llama3-8b-8192", model_provider="groq")
#print(llm)
from langchain_core.prompts import ChatPromptTemplate

system_message = """
Given an input question, create a syntactically correct {dialect} query to
run to help find the answer. Unless the user specifies in his question a
specific number of examples they wish to obtain, always limit your query to
at most {top_k} results. You can order the results by a relevant column to
return the most interesting examples in the database.

Never query for all the columns from a specific table, only ask for a the
few relevant columns given the question.

Pay attention to use only the column names that you can see in the schema
description. Be careful to not query for columns that do not exist. Also,
pay attention to which column is in which table.

Only use the following tables:
{table_info}
"""

user_prompt = """
Question: {input}

Supported operations:
- Filter by specific date or range (e.g., June 2025 or 2025-06-01 to 2025-06-30)
- Group data by week/month and get avg, min, max, close
- Get % change between two dates or months
"""

query_prompt_template = ChatPromptTemplate(
    [("system", system_message), ("user", user_prompt)]
)
'''
for message in query_prompt_template.messages:
    message.pretty_print()
'''
from typing_extensions import Annotated


class QueryOutput(TypedDict):
    """Generated SQL query."""

    query: Annotated[str, ..., "Syntactically valid SQL query."]


def write_query(state: State):
    """Generate SQL query to fetch information."""
    prompt = query_prompt_template.invoke(
        {
            "dialect": db.dialect,
            "top_k": 10,
            "table_info": db.get_table_info(),
            "input": state["question"],
        }
    )
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    return {"query": result["query"]}

#print(write_query({"question": "How many Records are there in HDFCBANK?"}))   
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool


def execute_query(state: State):
    """Execute SQL query."""
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    return {"result": execute_query_tool.invoke(state["query"])}

#print(execute_query({"query": "SELECT COUNT(*) as total from HDFCBANK_NS ;"}))
def generate_answer(state: State):
    """Answer question using retrieved information as context."""
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question.\n\n"
        f"Question: {state['question']}\n"
        f"SQL Query: {state['query']}\n"
        f"SQL Result: {state['result']}"
    )
    response = llm.invoke(prompt)
    return {"answer": response.content}

from langgraph.graph import START, StateGraph

graph_builder = StateGraph(State).add_sequence(
    [write_query, execute_query, generate_answer]
)
graph_builder.add_edge(START, "write_query")
graph = graph_builder.compile()

from IPython.display import Image, display
# Save graph as PNG file
# Get image bytes
'''
img_bytes = graph.get_graph().draw_mermaid_png()
# Save to a file
with open("my_langgraph.png", "wb") as f:
    f.write(img_bytes)

print("✅ Graph saved as my_langgraph.png. Open it to view.")


for step in graph.stream(
{"question": "How many records are there in TCS_NS?"},
stream_mode="updates"):
    print(step)
'''
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory, interrupt_before=["execute_query"])

# Now that we're using persistence, we need to specify a thread ID
# so that we can continue the run after review.
config = {"configurable": {"thread_id": "1"}}

for step in graph.stream(
    {"question": "Show the average close price for TCS_NS grouped by month?"},
    config,
    stream_mode="updates",
):
    print(step)

try:
    user_approval = input("Do you want to go to execute query? (y/n): ")
except Exception:
    user_approval = "n"

if user_approval.lower() in ["y", "yes"]:
    # If approved, continue the graph execution
    for step in graph.stream(None, config, stream_mode="updates"):
        print(step)
else:
    print("Operation cancelled by user.")