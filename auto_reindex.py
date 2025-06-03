import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from data_ingestion import load_and_split_docs
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VECTOR_DB_PATH = "./chroma"
FINANCE_DOCS_PATH = "./finance_docs"

# Reindex function
def reindex_documents():
    print("[INFO] Re-indexing started...")
    docs = load_and_split_docs(FINANCE_DOCS_PATH)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    if os.path.exists(VECTOR_DB_PATH):
        for file in os.listdir(VECTOR_DB_PATH):
            os.remove(os.path.join(VECTOR_DB_PATH, file))

    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=VECTOR_DB_PATH
    )
    vector_store.persist()
    print("[INFO] Re-indexing completed.")

# Watchdog handler
class FinanceDocsHandler(FileSystemEventHandler):
    def on_any_event(self, event):
        if event.is_directory:
            return
        if event.event_type in ("created", "modified", "deleted"):
            print(f"[WATCHDOG] Detected change: {event.event_type} - {event.src_path}")
            reindex_documents()

# Watcher setup
def start_watching():
    event_handler = FinanceDocsHandler()
    observer = Observer()
    observer.schedule(event_handler, path=FINANCE_DOCS_PATH, recursive=False)
    observer.start()
    print(f"[WATCHDOG] Watching folder: {FINANCE_DOCS_PATH}")

    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    start_watching()
