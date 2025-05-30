from fastapi import FastAPI
from pydantic import BaseModel
from rag_pipeline import qa_chain

app = FastAPI()

# Define input schema
class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(req: QueryRequest):
    try:
        response = qa_chain.run(req.question)
        return {"answer": response}
    except Exception as e:
        return {"error": str(e)}

# Health check endpoint (optional)
@app.get("/")
def root():
    return {"message": "RAG-powered Financial QA system is live."}


