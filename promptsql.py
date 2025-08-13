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

from langchain.chains.sql_database.prompt import SQL_PROMPTS

#print(list(SQL_PROMPTS))
from langchain.chat_models import init_chat_model

llm = init_chat_model("llama3-8b-8192", model_provider="groq")

from langchain.chains import create_sql_query_chain

chain = create_sql_query_chain(llm, db)
#chain.get_prompts()[0].pretty_print()
'''
context = db.get_context()
print(list(context))
print(context["table_info"])
'''
context = db.get_context()
prompt_with_context = chain.get_prompts()[0].partial(table_info=context["table_info"])
print(prompt_with_context.pretty_repr()[:1500])
