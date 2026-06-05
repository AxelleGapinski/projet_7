# "Puls-Events" : Chatbot RAG de recommandation culturelle

POC d'un chatbot RAG capable de répondre à des questions sur les événements culturels locaux en s'appuyant sur un système combinant LangChain, FAISS et Mistral.

---

## Architecture

```
-> Question utilisateur
-> Vectorisation (Qwen3-Embedding-0.6B)
-> Recherche FAISS → top-k chunks
-> Prompt LangChain → Mistral (small-latest)
-> Réponse + sources
```

## Structure du dépôt

```
projet_7/
├── scrap_clean_embed.py          # Scraping Open Agenda, nettoyage, chunking, embeddings
├── index_faiss.py    # Construction de l'index FAISS
├── rag_langchain.py    # Chaîne RAG : LangChain + Mistral
├── api.py      # API FastAPI 
├── tests/
│   ├── test_pipeline.py  # Tests unitaires
    └── api_test.py     # Tests fonctionnels de l'API
├── data/
│   └── events.json     # events scrappés et vectorisés
├── index/
│   ├── index.faiss     # Index vectoriel FAISS
│   └── index.pkl       # Métadonnées
├── .env                
├── .gitignore
├── requirements.txt
└── Dockerfile
```

---

## Installation

### Prérequis

- Python >= 3.10
- Docker
- Clé API Mistral

### Setup

```bash
# Cloner le dépôt
git clone <url-du-repo>
cd projet_7

# Créer l'environnement virtuel
python -m venv venv
venv\Scripts\activate  # windows
# source venv/bin/activate  # linux/Mac

# Installer les dépendances
pip install -r requirements.txt

# Configurer la clé API
cp .env.example .env
# Éditer .env et ajouter : MISTRAL_API_KEY=clé api
```

---

## Utilisation

### 1. Scraping et vectorisation des données

```bash
python scrap_clean_embed.py
```

Récupère les événements du département configuré (`DEPARTEMENT` dans `python scrap_clean_embed.py`), les nettoie, les découpe en chunks et génère les embeddings. Produit `data/events.json`

Pour changer de département, modifier la variable `DEPARTEMENT` dans `scrap_clean_embed.py` :
```python
DEPARTEMENT = "Indre"
```

### 2. Construction de l'index FAISS

```bash
python index_faiss.py
```

Construit l'index vectoriel LangChain depuis `data/events.json`. Retourne `index/index.faiss` et `index/index.pkl`

### 3. Lancer l'API

```bash
python api.py
```

L'API démarre sur `http://localhost:8000`. Documentation Swagger disponible sur `http://localhost:8000/docs`.

### 4. Tester l'API

```bash
python api_test.py
```

---

## Endpoints API

| Méthode | Route | Description |
|---------|-------|-------------|
| GET | `/` | Health check |
| POST | `/ask` | Pose une question, reçoit une réponse RAG |
| POST | `/rebuild` | Reconstruit l'index FAISS |
| GET | `/docs` | Documentation Swagger interactive |

### Exemple d'appel

```python
import requests

response = requests.post("http://localhost:8000/ask", json={
    "question": "y a-t-il des concerts ce mois-ci?",
    "top_k": 5
})
print(response.json())
```

Réponse :
```json
{
  "question": "y a-t-il des concerts ce mois-ci?",
  "answer": "Oui, voici les concerts disponibles : ......",
  "sources": [
    {
      "title": "Concert de jazz",
      "location": "Salle des fêtes, Toulouse",
      "date": "2026-06-15",
      "url": "https://..."
    }
  ]
}
```

---

## Docker

```bash
# 1. Construire l'image (après avoir généré data/ et index/)
docker build -t puls-events-api .

# 2. Lancer le conteneur
docker run -p 8000:8000 -e MISTRAL_API_KEY=clé puls-events-api

# 3. Tester sur http://localhost:8000/docs
```

---

## Tests

```bash
# tests unitaires
pytest tests/test_pipeline.py -v

# tests fonctionnels
python api_test.py
```

---

## Modèles utilisés

| Rôle | Modèle | Justification |
|------|--------|---------------|
| Embeddings | Qwen3-Embedding-0.6B | Gratuit, open source, multilingue|
| LLM | mistral-small-latest | API gratuite |
| Vectorstore | FAISS IndexFlatIP | Produit scalaire = plus simple |
| Orchestration | LangChain | Standard, facilite la piepline RAG |

---

## Variables d'environnement

| Variable | Description | Obligatoire |
|----------|-------------|-------------|
| `MISTRAL_API_KEY` | Clé API Mistral (console.mistral.ai) | Oui |