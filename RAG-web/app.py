import os
import pathlib
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
load_dotenv()

current_directory = pathlib.Path.cwd()
persistent_directory = os.path.join(current_directory, "db", "chromadb_webngit")

embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

db = Chroma(persist_directory=persistent_directory, embedding_function=embedder)

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k":3},
)

prompt_template = PromptTemplate(
                    input_variables=["context", "question"],
                    template="""
                    You are an expert in analyzing and answering questions about current news.

                    Use the following extracted context to answer the question.
                    If the answer isn't directly available, say "Not mentioned in the context."

                    Context:
                    {context}

                    Question: {input}
                    Answer:
                    """
                    )

question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
chain = create_retrieval_chain(retriever, question_answer_chain)

question = "What is qlib? What problem does it solve"
response = chain.invoke({"input": question})
print(response['answer'])


