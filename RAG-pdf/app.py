import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_google_genai  import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5},
)

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

### --- Contextualization Prompt (Step 1) ---
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "Given a chat history and the latest user question "
     "which might reference context in the chat history, "
     "formulate a standalone question that can be understood "
     "without the chat history. Do NOT answer the question, just return it."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# Create a contextual retriever
history_aware_retriever = create_history_aware_retriever(
    llm=llm,
    retriever=retriever,
    prompt=contextualize_q_prompt
)

### --- Answering Prompt (Step 2) ---
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a helpful assistant. Use the following retrieved context "
     "to answer the user's question. If unsure, say 'I don't know'. "
     "Be detailed. \n\n{context}"), 
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# Create the RAG chain
question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


# Function to simulate a continual chat
def continual_chat():
    print("Start chatting with the AI! Type 'exit' to end the conversation.")
    chat_history = []  # Collect chat history here (a sequence of messages)
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break
        # Process the user's query through the retrieval chain
        result = rag_chain.invoke({"input": query, "chat_history": chat_history})
        # Display the AI's response
        print(f"AI: {result['answer']}")
        # Update the chat history
        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=result["answer"]))

# Main function to start the continual chat
if __name__ == "__main__":
    continual_chat()
