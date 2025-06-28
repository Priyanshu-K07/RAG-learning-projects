import os
import pathlib

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

current_directory = pathlib.Path.cwd()
bills_directory = os.path.join(current_directory, "bills")
persistent_directory = os.path.join(current_directory, "db", "chromadb")

if not os.path.exists(persistent_directory):
    print("Vector store do not exist. Initializing...")
    all_bills = [f for f in os.listdir(bills_directory) if f.endswith(".pdf")]

    print("Loading data...")
    documents = []
    for bill in all_bills:
        file_path = os.path.join(bills_directory, bill)
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata = { "source":bill }
            documents.append(doc)

    print("Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # Keeps each job/edu section mostly intact
        chunk_overlap=50,    # Avoids cutting off skills or responsibilities
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)

    print("Embedding...")
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
        persist_directory=persistent_directory,
    )

    print("Done.")

else:
    print("Vector store already exists. No need to initialize.")
    

    

    





