import json
import pickle
import pandas as pd
from pathlib import Path
from langchain_community.vectorstores import FAISS as LangchainFAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

DATA_DIR = Path("data")
INDEX_DIR = Path("index")

# le modèle d'embedding chargé une seule fois au démarrage du script
embedder = HuggingFaceEmbeddings(
    model_name="Qwen/Qwen3-Embedding-0.6B",
    model_kwargs={"trust_remote_code": True},
    encode_kwargs={"normalize_embeddings": True}
)


def build_index(json_path: str = DATA_DIR / "events.json"):
    """
    Construit l'index FAISS au format LangChain depuis events.json.
    à relancer si les données changent.
    """

    INDEX_DIR.mkdir(exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    # on convertit chaque chunk en Document LangChain
    # page_content = le texte qui sera vectorisé
    # metadata = les infos qu'on veut récupérer dans les résultats
    documents = [
        Document(
            page_content=r["text_for_embedding"],
            metadata={
                "title": r["title"],
                "location": r["location"],
                "first_date": r["first_date"],
                "url": r["url"],
                "description": r["description"],
            }
        )
        for r in records
    ]

    # LangChain vectorise les documents et construit l'index FAISS en une ligne
    print("Début de construction de l'index FAISS avec LangChain")
    vectorstore = LangchainFAISS.from_documents(documents, embedder)
    print("FAISS ok")

    # sauvegarde sur disque (crée index.faiss + index.pkl dans index/)
    vectorstore.save_local(str(INDEX_DIR))
    print("index sauvegardé sur le disque")

    print(f"Index créé : {len(documents)} chunks")
    return vectorstore


def load_index():
    """
    Charge l'index depuis le disque.
    À utiliser dans l'API pour ne pas reconstruire l'index à chaque requête.
    """
    return LangchainFAISS.load_local(
        str(INDEX_DIR),
        embedder,
        allow_dangerous_deserialization=True
    )


def search(query: str, vectorstore, top_k: int = 5):
    """
    Recherche les chunks les plus similaires à une question texte.
    Prend directement le texte de la question (plus besoin de vectoriser manuellement).

    Returns: liste de dicts avec score + métadonnées
    """
    # similarity_search_with_score retourne des tuples (Document, score)
    results = vectorstore.similarity_search_with_score(query, k=top_k)

    return [
        {
            "score": float(score),
            "title": doc.metadata.get("title", ""),
            "location": doc.metadata.get("location", ""),
            "first_date": doc.metadata.get("first_date", ""),
            "url": doc.metadata.get("url", ""),
            "description": doc.metadata.get("description", ""),
            "text": doc.page_content,
        }
        for doc, score in results
    ]


if __name__ == "__main__":
    build_index()