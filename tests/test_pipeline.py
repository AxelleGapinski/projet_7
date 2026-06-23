###tests unitaires

import json
import numpy as np
from pathlib import Path


def test_events_json_valide():
    """Les données existent ne sont pas vides et ont les bons champs"""
    assert Path("data/events.json").exists(), "data/events.json manquant"
    with open("data/events.json", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) > 0, "Aucun événement"
    for field in ["uid", "title", "text_for_embedding", "embedding"]:
        assert field in data[0], f"Champ '{field}' manquant"


def test_embeddings_normalises():
    """Les vecteurs sont normalisés"""
    with open("data/events.json", encoding="utf-8") as f:
        data = json.load(f)
    for event in data[:5]:
        norm = np.linalg.norm(event["embedding"])
        assert abs(norm - 1.0) < 0.01, f"vecteur non normalisé (norme={norm:.3f})"


def test_index_faiss():
    """L'index faiss se charge et retourne bien des résultats structurés"""
    from index_faiss import load_index, search
    assert Path("index/index.faiss").exists(), "Index manquant"
    vectorstore = load_index()
    results = search("concert musique", vectorstore, top_k=3)
    assert len(results) > 0
    for key in ["score", "title", "location", "first_date", "url"]:
        assert key in results[0], f"Clé '{key}' manquante"


def test_retourne_reponse():
    """retourne une réponse non vide avec les bonnes clés"""
    from rag_langchain import ask
    result = ask("y a-t-il des concerts à Bordeaux?", top_k=3)
    assert "question" in result and "answer" in result and "sources" in result
    assert len(result["answer"]) > 10


def test_hors_sujet():
    """Une question hors sujet ne fait pas planter le rag"""
    from rag_langchain import ask
    result = ask("Quel est le prix du pétrole ?", top_k=3)
    assert len(result["answer"]) > 0
