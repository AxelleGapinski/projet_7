import json
import logging
import re
from datetime import datetime, timedelta
from html import unescape
from pathlib import Path
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer

### CONFIG

# configuration des logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# URL API
BASE_URL = (
    "https://public.opendatasoft.com/api/explore/v2.1"
    "/catalog/datasets/evenements-publics-openagenda/records"
)

# département ciblé
DEPARTEMENT = "Gironde"

# Dossier de sauvegarde
DATA_DIR = Path("data")

# Modèle embeddings
# QWEN car gratuit et bon en multilingue
MODEL = SentenceTransformer(
    "Qwen/Qwen3-Embedding-0.6B",
    trust_remote_code=True
)


### COLLECTE DONNEES
def fetch_events(department: str = DEPARTEMENT) -> list[dict]:
    """
    Récupère les événements d’un département défini via l’API

    Args: department (str): département ciblé

    Returns: list[dict]: liste des événements récupérés
    """

    # date limite: événements des 12 derniers mois
    date_limit = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    # Filtre utilisé dans la requête API
    where = (
        f'location_department="{department}" '
        f'AND lastdate_begin>="{date_limit}"'
    )

    all_records = []
    offset = 0

    while True:

        data = requests.get(
            BASE_URL,
            params={
                "where": where,
                "limit": 100,
                "offset": offset
            },
            timeout=30
        ).json()

        results = data.get("results", [])

        # arrêt si aucun résultat
        if not results:
            break

        # ajout des résultats
        all_records.extend(results)

        offset += 100

        logger.info(
            "Récupérés: %d / %s",
            len(all_records),
            data.get("total_count", "?")
        )

        # Arrêt si tous les événements ont été récupérés
        if len(all_records) >= data.get("total_count", 0):
            break

    return all_records


## NETTOYAGE DES DONNEES

def clean_html(text: str | None) -> str:
    """
    Nettoie le texte HTML:
    - supprimr les balises HTML
    - convertir les caractères spéciaux
    - supprimer espaces multiples

    Args: text (str | None): Texte brut

    Returns: str: Texte nettoyé
    """

    if not text:
        return ""

    # Suppression des balises HTML
    text = re.sub(r"<[^>]+>", " ", text)

    # Conversion des caractères HTML spéciaux
    text = unescape(text)

    # suppression des espaces multiples
    text = re.sub(r"\s+", " ", text)

    return text.strip()

### STRUCTURATION DES EVENEMENTS
def parse(record: dict) -> dict:
    """
    Transforme un événement brut en structuré

    Args:
        record (dict): Événement brut provenant de l’API

    Returns:
        dict: événement nettoyé et structuré
    """

    # coordonnées géographiques
    coords = record.get("location_coordinates") or {}

    # Description nettoyée
    desc = (
        clean_html(record.get("longdescription_fr"))
        or clean_html(record.get("description_fr"))
    )

    # Titre
    title = record.get("title_fr", "")

    # Lieu
    location = ", ".join(
        filter(
            None,
            [
                record.get("location_name"),
                record.get("location_city")
            ]
        )
    )

    # Mots-clés
    keywords = ", ".join(record.get("keywords_fr") or [])

    # texte concaténé utilisé comme entrée pour être transformé en embeddings
    text = "\n".join(
        filter(
            None,
            [
                f"Titre : {title}",
                f"Date : {record.get('daterange_fr', '')}",
                f"Lieu : {location}" if location else "",
                f"Description: {desc}" if desc else "",
                f"Mots-clés : {keywords}" if keywords else "",
            ]
        )
    )

    return {
        "uid": record.get("uid"),
        "title": title,
        "description": desc,
        "keywords": keywords,
        "first_date": record.get("firstdate_begin"),
        "last_date": record.get("lastdate_begin"),
        "location": location,
        "latitude": coords.get("lat"),
        "longitude": coords.get("lon"),
        "url": record.get("canonicalurl"),

        # texte envoyé au modèle pour créer l’embedding
        "text_for_embedding": text,
    }

### NETTOYAGE DES EVENEMENTS
def clean_events(raw: list[dict]) -> pd.DataFrame:
    """
    Nettoie et filtre les événements:
    - conversion en dataframe
    - suppression des doublons
    - suppression des textes trop courts

    Args:
        raw (list[dict]): liste des événements bruts

    Returns:
        pd.DataFrame: df nettoyé
    """

    # parsing des événements
    df = pd.DataFrame([parse(r) for r in raw])

    # suppression doublons
    df = df.drop_duplicates("uid")

    # Suppression textes trop courts
    df = df[df["text_for_embedding"].str.len() >= 30]

    # Réindexation
    df = df.reset_index(drop=True)

    logger.info("%d événements gardés", len(df))

    return df

## CHUNKING
def chunk_text(text: str, max_chars: int = 500, overlap: int = 50) -> list[str]:
    """
    Découpe en chunks avec overlap

    Args:
        text: texte à découper
        max_chars: Taille max chunk (en caracteres)
        overlap: taille overlap entre chunks

    Returns:
        list[str]: liste de chunks
    """
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + max_chars

        # couper sur un espace si possible
        if end < len(text):
            cut = text.rfind(" ", start, end)
            if cut != -1:
                end = cut

        chunks.append(text[start:end].strip())
        start = end - overlap

    return chunks


def chunk_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Divise chaque événement en plusieurs lignes si le texte est long:

    Args:
        df: df avec colonne text_for_embedding

    Returns:
        pd.DataFrame: df avec une ligne par chunk
    """
    rows = []

    for _, row in df.iterrows():
        chunks = chunk_text(row["text_for_embedding"])

        for i, chunk in enumerate(chunks):
            new_row = row.to_dict()
            new_row["chunk_index"] = i
            new_row["chunk_total"] = len(chunks)
            new_row["text_for_embedding"] = chunk
            rows.append(new_row)

    df_chunked = pd.DataFrame(rows).reset_index(drop=True)
    logger.info(
        "Chunking : %d événements → %d chunks",
        len(df), len(df_chunked)
    )
    return df_chunked


## GENERATION DES EMBEDDINGS
def vectorize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Génère les embeddings des événements

    Args:
        df (pd.DataFrame): df contenant les textes

    Returns:
        pd.DataFrame: df avec une colonne embedding ajoutée
    """

    logger.info("Vectorisation lancée avec Qwen3-Embedding.........")

    # génération des vecteurs embeddings
    embeddings = MODEL.encode(
        df["text_for_embedding"].tolist(),

        # nb de textes traités simultanément
        batch_size=32,

        # Normalisation des vecteurs
        normalize_embeddings=True,

        # Affichage progression
        show_progress_bar=True,
    )

    # Ajout des embeddings au df
    df["embedding"] = embeddings.tolist()

    return df


##  SAUVEGARDE
def save(df: pd.DataFrame):
    """
    Sauvegarde les événements en CSV et JSON

    Args:
        df (pd.DataFrame): dataframe des événements
    """

    DATA_DIR.mkdir(exist_ok=True)

    # Sauvegarde CSV (sans les embeddings)
    df.drop(columns=["embedding"]).to_csv(
        DATA_DIR / "events.csv",
        index=False
    )

    # sauvegarde JSON
    with open(DATA_DIR / "events.json", "w", encoding="utf-8") as f:

        json.dump(
            df.to_dict(orient="records"),
            f,
            ensure_ascii=False,
            indent=2
        )

    logger.info("sauvegardé dans %s", DATA_DIR)


## PIPELINE
if __name__ == "__main__":

    # Récupération des événements
    raw = fetch_events()[:5000] # modif, 100 premiers pour test

    # Nettoyage et structuration des events
    df = clean_events(raw)

    # Chunking
    df = chunk_events(df)

    #Génération des embeddings
    df = vectorize(df)

    # Sauvegarde
    save(df)