import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from data_ingestion import load_and_split_docs

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VECTOR_DB_PATH = "./chroma"

# Step 1: Load and split documents
docs = load_and_split_docs("./finance_docs")

# Step 2: Initialize embedding model
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Step 3: Create or load Chroma vector store
vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory=VECTOR_DB_PATH
)

# Step 4: Set up retriever
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Step 5: Initialize LLM
llm = ChatOpenAI(model_name="gpt-4", temperature=0.2, openai_api_key=OPENAI_API_KEY)

# Step 6: Define a prompt template for the QA chain
prompt_template = PromptTemplate.from_template(
    "Answer the following question based on the provided context:\n\n{context}\n\nQuestion: {input}\n\nAnswer:"
)

# Step 7: Create the document chain to stuff documents into the prompt
document_chain = create_stuff_documents_chain(llm, prompt_template)

# Step 8: Create the retrieval chain
qa_chain = create_retrieval_chain(retriever, document_chain)

# Step 9: Test query (Optional)
if __name__ == "__main__":
    query = "What are the key FX risk management strategies?"
    result = qa_chain.invoke({"input": query})
    print("\nAnswer:\n", result["answer"])
    print("\nSource Documents:\n", result["context"])



