from llama_index.core import SimpleDirectoryReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_split_docs(doc_path="./finance_docs"):
    documents = SimpleDirectoryReader(input_dir=doc_path).load_data()

    texts = [doc.text for doc in documents]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.create_documents(texts)

    return chunks

if __name__ == "__main__":
    docs = load_and_split_docs()
    print(f"Loaded and split {len(docs)} chunks.")