import os
import pathlib

from pdf2image import convert_from_path
import pytesseract

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

current_directory = pathlib.Path.cwd()
books_directory = os.path.join(current_directory, "books")
persistent_directory = os.path.join(current_directory, "db", "chromadb")

def extract_text_from_scanned_pdf(pdf_path):
    images = convert_from_path(pdf_path, poppler_path=r"C:\poppler-24.08.0\Library\bin")
    full_text = ""
    for page_img in images:
        text = pytesseract.image_to_string(page_img)
        full_text += text + "\n"
    return full_text

def ocr_to_documents(pdf_path):
    text = extract_text_from_scanned_pdf(pdf_path)
    return [Document(page_content=text)]


if not os.path.exists(persistent_directory):
    print("Vector store do not exist. Initializing...")
    all_books = [f for f in os.listdir(books_directory) if f.endswith(".pdf")]

    print("Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      
        chunk_overlap=200,    
    )
    documents = []
    for book in all_books:
        file_path = os.path.join(books_directory, book)
        documents.extend(ocr_to_documents(file_path))
        print(f"{book} done.")
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
