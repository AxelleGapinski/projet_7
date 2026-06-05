import os
import numpy as np
from mistralai.client.sdk import Mistral
from sentence_transformers import SentenceTransformer
from index_faiss import load_index, search
from dotenv import load_dotenv
load_dotenv()

# configs Mistral
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=MISTRAL_API_KEY)
MISTRAL_MODEL = "mistral-small-latest"

# chargement modèle d'embeddings QWEN
embedder = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True)

# chargement de l'index Faiss + métadonnées
index, metadata = load_index()

# fonction principale
def ask(question: str, top_k: int = 5) -> dict:
    """
    Fonction appelée depuis l'API,qui prend une question et retourne une réponse
    top_k = combien d'événements on veut récupérer dans FAISS pour construire le contexte envoyé à Mistral
    """

    # transformer la question utilisateur en vecteurs + cherche les évènement simimlaires dans l'index 
    query_embedding = embedder.encode(question, normalize_embeddings=True)
    chunks = search(query_embedding, index, metadata, top_k=top_k)

    # construire le contexte obtenu pour le renvoyer à mistral
    # formattage plus propre
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        part = f"""Événement {i} :
- Titre: {chunk['title']}
- Lieu: {chunk['location']}
- Date: {chunk['first_date']}
- Description: {chunk['description'][:2000]}
- Lien : {chunk['url']}"""
        context_parts.append(part)

    context = "\n\n".join(context_parts)

    # appel à mistral
    response = client.chat.complete(
        model=MISTRAL_MODEL,
        messages=[
            {
                # system prompt
                "role": "system",
                "content": (
                    "Tu es un assistant culturel spécialisé dans les événements locaux et u aides les utilisateurs à trouver des événements culturels dans leur région. Réponds uniquement à partir des événements fournis et si aucun événement ne correspond à la question, dis  le clairement et n'invente pas de réponse. Réponds toujours en français de manière concise"
                )
            },
            {
                # user prompt
                "role": "user",
                "content": (
                    f"voici les événements disponibles :\n\n{context}"
                    f"\n\nQuestion : {question}"
                )
            }
        ]
    )

    answer = response.choices[0].message.content

    #retourne un dict propre avec la question de base, la réponse générée par le LLM, les sources utilisées (les chunks faiss)
    return {
        "question": question,
        "answer": answer,
        "sources": [
            {
                "title": c["title"],
                "location": c["location"],
                "date": c["first_date"],
                "score": round(c["score"], 3),  #score arrondi 
                "url": c["url"]
            }
            for c in chunks
        ]
    }


# pour tester le script directement avec python rag.py sans passer par l'API
if __name__ == "__main__":
    result = ask("est-ce qu'il ya des expositions ou des concerts en octobre?")

    print("Réponse :", result["answer"])
    print("\nSources utilisées :")
    for s in result["sources"]:
        print(f"  - {s['title']} | {s['location']} | score: {s['score']}")