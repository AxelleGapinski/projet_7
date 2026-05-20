import json
import pickle
import numpy as np
import faiss
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
INDEX_DIR = Path("index")

# construction de l'index
def build_index(json_path: str = DATA_DIR / "events.json"):
    """
    Charge les événements vectorisés dans les events.json et construit l'index FAISS + le fichier de métadonnées associé
    à relancer si les données changent
    """

    # création du dossier d'index
    INDEX_DIR.mkdir(exist_ok=True)

    # charger les données
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    df = pd.DataFrame(records)

    # extraire les vecteurs qu'on avait généré avec QWEN
    embeddings = np.array(df["embedding"].tolist(), dtype="float32") # on convertit en numpy (float32 nécessaire pour FAISS)
    dim = embeddings.shape[1] 

    # créer l'index FAISS
    index = faiss.IndexFlatIP(dim) # recherche par similarité cosinus

    # ajout des vecteurs dans l'index
    index.add(embeddings)

    #sauvegarder l'index
    faiss.write_index(index, str(INDEX_DIR / "events.faiss"))

    # sauvegarder les métadonnées
    metadata = df[["uid", "title", "description", "keywords", "first_date", "last_date", "location", "latitude", "longitude", "url", "chunk_index", "chunk_total", "text_for_embedding"]].to_dict(orient="records")

    with open(INDEX_DIR / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print(f"Index FAISS créé = {len(metadata)} chunks, dim={dim}")
    return index, metadata


def load_index():
    """Charge l'index FAISS et les métadonnées.
    à utiliser dans l'API pour pas reconstruire l'index à chaque requête faite
    """

    # lecture de l'index et des métadonnées
    index = faiss.read_index(str(INDEX_DIR / "events.faiss"))
    with open(INDEX_DIR / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return index, metadata


def search(query_embedding: np.ndarray, index, metadata, top_k: int = 5):
    """
    Recherche les chunks les plus similaires au vecteurs de la requête faite
    Args : 
        query_embedding : le vecteur de la requête
        index : l'index faiss
        metadata : les métadonnées
        top_k : le nb de résultats à retourner
    Returns: liste de dicts avec pour chaque le score de similarité + les métadonnées du chunk
    """

    # convertit la requête en numpy array
    query = np.array([query_embedding], dtype="float32")

    # recherche dans l'index
    scores, indices = index.search(query, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append({
            "score": float(score),
            **metadata[idx]
        })
    return results

# construit l'index
if __name__ == "__main__":
    build_index()

