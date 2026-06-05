from fastapi import FastAPI
from pydantic import BaseModel
from rag import ask

#création de l'app fastAPI
app = FastAPI(title="RAG API")

class Question(BaseModel):
    question: str  
    top_k: int = 5

@app.post("/ask")
def ask_question(body: Question):
    return ask(body.question, top_k=body.top_k)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)