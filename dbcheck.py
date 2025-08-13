'''
from sqlalchemy import create_engine
import pandas as pd

engine = create_engine("sqlite:///db/stock_data.db")

df = pd.read_sql("SELECT Date, Close FROM TCS_NS ORDER BY Date DESC LIMIT 1;", engine)
print(df)
'''

from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri("sqlite:///db/stock_data.db")
print(db.dialect)
print(db.get_usable_table_names())
print(db.run("SELECT * FROM TCS_NS LIMIT 1;"))