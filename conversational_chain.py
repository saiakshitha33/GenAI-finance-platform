from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

from data_ingestion import load_and_split_docs
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Step 1: Load and split documents
docs = load_and_split_docs("./finance_docs")

# Step 2: Initialize embedding model
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Step 3: Create or load Chroma vector store
VECTOR_DB_PATH = "./chroma"
vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory=VECTOR_DB_PATH
)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Step 4: Memory + Chat Model
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
llm = ChatOpenAI(model_name="gpt-4", temperature=0.2, openai_api_key=OPENAI_API_KEY)

# Step 5: Conversational Chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    output_key="answer"  # ✅ Specify which key to store in memory
)

# ✅ Expose chain for FastAPI
conversational_chain = qa_chain

# Step 6: Example query
def run_example():
    query1 = "What are the key FX risk management strategies?"
    print("\nQ:", query1)
    result1 = qa_chain.invoke({"question": query1})
    print("A:", result1["answer"])

    query2 = "Explain what forward contracts are in this context."
    print("\nQ:", query2)
    result2 = qa_chain.invoke({"question": query2})
    print("A:", result2["answer"])

if __name__ == "__main__":
    run_example()
