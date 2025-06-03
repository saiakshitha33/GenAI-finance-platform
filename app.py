from fastapi import FastAPI
from pydantic import BaseModel
from conversational_chain import conversational_chain  # ✅ New chain with memory
import csv
from datetime import datetime
from fastapi.responses import JSONResponse

app = FastAPI()

# Define input schema
class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(req: QueryRequest):
    try:
        result = conversational_chain.invoke({"question": req.question})  # ✅ Correct key is 'question'

        # Log the query
        with open("query_logs.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                req.question,
                result.get("answer", ""),
                " | ".join([m.content for m in result.get("chat_history", []) if hasattr(m, "content")])
            ])

        return {
            "question": req.question,
            "answer": result.get("answer", "No answer found."),
            "chat_history": [m.content for m in result.get("chat_history", []) if hasattr(m, "content")]
        }

    except Exception as e:
        return {"error": str(e)}


@app.get("/logs")
def get_logs():
    try:
        logs = []
        with open("query_logs.csv", "r") as f:
            reader = csv.reader(f)
            for row in reader:
                logs.append({
                    "timestamp": row[0],
                    "question": row[1],
                    "answer": row[2],
                    "chat_summary": row[3] if len(row) > 3 else ""
                })
        return JSONResponse(content=logs)
    except FileNotFoundError:
        return {"message": "No logs found yet."}
    except Exception as e:
        return {"error": str(e)}


@app.get("/")
def root():
    return {"message": "RAG-powered Financial QA system with conversational memory is live."}
