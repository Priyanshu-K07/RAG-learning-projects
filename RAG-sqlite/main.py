import os
import pathlib
from dotenv import load_dotenv

from langchain.chains import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from sqlalchemy import create_engine, text

load_dotenv()

current_directory = pathlib.Path.cwd()
db_path = os.path.join(current_directory, "data", "IMDB.sqlite")

db = SQLDatabase.from_uri(f"sqlite:///{db_path}")

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

db_chain = create_sql_query_chain(llm=llm, db=db)

query = "Show me top 5 highest worldwide grossing movies and their genres. If multiple genre exist, make them comma separated in same cell"
sql_response = db_chain.invoke({"question": query})

cleaned_query = sql_response.strip().replace("```sqlite","").replace("```","").strip()
print("Generated SQL:\n", cleaned_query)

engine = create_engine(f"sqlite:///{db_path}")
with engine.connect() as conn:
    result = conn.execute(text(cleaned_query))
    rows = result.fetchall()
    for row in rows:
        print(row)