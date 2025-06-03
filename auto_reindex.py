import os
import shutil
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from langchain_openai import OpenAIEmbeddings  # Updated import
from langchain_chroma import Chroma
from data_ingestion import load_and_split_docs
from dotenv import load_dotenv
import logging
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VECTOR_DB_PATH = "./chroma"

# Re-index documents
def reindex_documents():
    logging.info("Re-indexing started...")
    try:
        # Load and split documents
        docs = load_and_split_docs("./finance_docs")
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

        # Clear existing Chroma database
        if os.path.exists(VECTOR_DB_PATH):
            try:
                # Attempt to delete the collection
                vector_store = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)
                vector_store.delete_collection()  # Delete the default collection
                logging.info("Cleared existing Chroma collection.")
            except Exception as e:
                logging.warning(f"Failed to delete collection: {str(e)}. Falling back to directory deletion.")
                # Fallback: Delete the entire directory
                shutil.rmtree(VECTOR_DB_PATH, ignore_errors=True)
                os.makedirs(VECTOR_DB_PATH, exist_ok=True)

        # Re-index documents
        Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=VECTOR_DB_PATH
        )
        logging.info("Re-indexing completed.")
    except Exception as e:
        logging.error(f"Re-indexing failed: {str(e)}")

# Watchdog event handler
class FileChangeHandler(FileSystemEventHandler):
    def on_any_event(self, event):
        if event.is_directory:
            return
        logging.info(f"Detected change: {event.event_type} - {event.src_path}")
        # Add a small delay to ensure file operations are complete
        time.sleep(1)
        reindex_documents()

# Watchdog observer
def start_watching():
    observer = Observer()
    observer.schedule(FileChangeHandler(), path="./finance_docs", recursive=True)
    observer.start()
    logging.info("Watching folder: ./finance_docs")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    start_watching()