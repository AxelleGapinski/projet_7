import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from rag_langchain import ask
from index_faiss import build_index


load_dotenv()

app = FastAPI(
    title="Puls-Events RAG API",
    description="API de recommandation d'événements culturels basée sur un système RAG",
    version="1.0.0"
)

class Question(BaseModel):
    question: str
    top_k: int = 5

class RebuildResponse(BaseModel):
    status: str
    message: str

@app.get("/")
def root():
    """Vérifie que l'API tourne bien"""
    return {"status": "ok", "message": "RAG API is running"}


@app.post("/ask")
def ask_question(body: Question):
    """
    Pose une question au chatbot RAG et retourne une réponse générée par Mistral + les sources utilisées
    """
    # Validation : question non vide
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="La question ne peut pas être vide.")

    try:
        result = ask(body.question, top_k=body.top_k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur RAG : {str(e)}")


@app.post("/rebuild")
def rebuild_index():
    """
    Reconstruit la base vectorielle FAISS depuis les données existantes.
    À appeler après un nouveau scraping des events
    """
    try:
        build_langchain_index()
        return RebuildResponse(
            status="ok",
            message="Index FAISS reconstruit"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur rebuild : {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)