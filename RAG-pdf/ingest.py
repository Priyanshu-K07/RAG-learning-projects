import os

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

current_dir = os.path.dirname(os.path.abspath(__file__))
books_dir = os.path.join(current_dir, "books")
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")
    
    if not os.path.exists(books_dir):
        raise FileNotFoundError(
            f"The directory {books_dir} does not exist. Please check the path."
        )

    # List all text files in the directory
    book_files = [f for f in os.listdir(books_dir) if f.endswith(".pdf")]

    print("loading data...")
    documents = []
    for book_file in book_files:
        file_path = os.path.join(books_dir, book_file)
        loader = PyMuPDFLoader(file_path)
        book_docs = loader.load()
        for doc in book_docs:
            # Add metadata to each document indicating its source
            doc.metadata = {"source": book_file}
            documents.append(doc)

    print("splitting documents...")
    token_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = token_splitter.split_documents(documents)

    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("embedding into vector store...")
    db = Chroma(
    embedding_function=embedder,
    persist_directory=persistent_directory,
    )

    batch_size = 5000
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        db.add_documents(batch)

    print("Done.")

else:
    print("Vector store already exists. No need to initialize.")
    
    

    
