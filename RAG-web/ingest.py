import os
import pathlib

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders.git import GitLoader

current_directory = pathlib.Path.cwd()
persistent_directory = os.path.join(current_directory, "db", "chromadb_webngit")


urls = [
    "https://indianexpress.com",
    "https://timesofindia.indiatimes.com",
    "https://www.ndtv.com",
    "https://www.hindustantimes.com"
    # Add more URLs as needed
]

github_repo = "https://github.com/microsoft/qlib.git"

def git_loader(repo):
    loader = GitLoader(repo_path="./qlib-local", clone_url=repo)
    docs = loader.load()
    return docs

def load_multiple_urls(urls):
    all_documents = []
    for url in urls:
        loader = WebBaseLoader(url)
        documents = loader.load()  
        all_documents.extend(documents)  
    return all_documents


if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    print("Loading data...")
    web = load_multiple_urls(urls=urls)
    git = git_loader(repo=github_repo)

    docs = web + git

    print("Splitting...")
    textsplitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = textsplitter.split_documents(docs)

    print("Embedding into vector store...")
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    db = Chroma.from_documents(documents=chunks,
                               embedding=embedder,
                               persist_directory=persistent_directory)
    
    print("Done.")

else:
    print("Vector store already exists.")