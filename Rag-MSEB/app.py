import os
import pathlib
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate

load_dotenv()

current_directory = pathlib.Path.cwd()
persistent_directory = os.path.join(current_directory, "db", "chromadb")

embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(embedding_function=embedder, persist_directory=persistent_directory)

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

prompt = PromptTemplate.from_template("""
You are a light bill analysis assistant.

Given the following electricity bill content, extract the required information and answer the user's query.

Context:
{context}

Question: {input}

Answer:
Provide the following structured details if available:
- Consumer Number
- Billing Name
- Address
- Billing Amount
- Due Date

The user can provide either the Consumer Number or the Billing Name. Ensure the output is structured and easy to read.
""")

qa_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)

query = "Individual Tenant"
response = rag_chain.invoke({"input": query})
print(response['answer'])