# "Puls-Events" : Chatbot RAG de recommandation culturelle

POC d'un chatbot RAG capable de répondre à des questions sur les événements culturels locaux en s'appuyant sur un système combinant LangChain, FAISS et Mistral.

---

## Table des matières
 
- [Installation](#installation)
- [Endpoints API](#endpoints-api)
- [Docker](#docker)
- [Tests](#tests)
- [Le projet](#le-projet)
- [Données](#données)
- [Modèles & choix techniques](#modèles)
- [Base vectorielle FAISS](#base-vectorielle-faiss)
- [Évaluation du système](#evaluation-du-système)
- [Bilan et perspectives](#bilan--perspectives)

---
# Partie 1 : Installation

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

## Structure du repository

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

## Endpoints API

| Méthode | Route | Description |
|---------|-------|-------------|
| GET | `/` | Health check |
| POST | `/ask` | Pose une question, reçoit une réponse RAG |
| POST | `/rebuild` | Reconstruit l'index FAISS |
| GET | `/docs` | Documentation Swagger interactive |

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


# Partie 2 : Le projet


## Objectifs du projet
POC d'un chatbot intelligent capable de répondre à des questions sur les événements culturels locaux, en s'appuyant sur un système RAG (Retrieval-Augmented Generation) combinant LangChain, FAISS et Mistral.
> Mission : démonstration de faisabilité technique avant intégration produit.

---

## Architecture de la pipeline 

```
-> Question utilisateur - Y a til des concerts ce weekend ?
-> Vectorisation (Qwen3-Embedding-0.6B) - La question est transformée en vecteurs
-> Recherche FAISS → top-k chunks - Les top-k évènements aux vecteurs les + similaires sont récupérés dans la DB
-> Prompt LangChain → Mistral (small-latest) - Le LLM prompté reçoit les top-k évènements et la question
-> Réponse + sources - Le LLM formule une réponse et fournit les sources
```

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

## Données

**Source** : API publique Open Agenda (`public.opendatasoft.com`)
 
**Filtres appliqués** :
- Département : Gironde (configurable)
- Période : événements des 12 derniers mois + à venir

**Pipeline de traitement** (`scrap_clean_embed.py`) :
 
1. **Scraping** : récupération paginée via l'API, 100 événements par requête
2. **Nettoyage** : suppression des balises HTML, conversion des caractères spéciaux, suppression des espaces multiples, dédoublonnage
3. **Structuration** : chaque événement est mis en forme en texte concaténé (titre + date + lieu + description + mots-clés) pour l'embedding
4. **Chunking** : les textes longs sont découpés en chunks de 500 car avec un overlap de 50 caractères (pour ne pas perdre de contexte entre deux chunks)
5. **Embedding** : chaque chunk est vectorisé avec Qwen3-Embedding-0.6B, les vecteurs sont normalisés (norme = 1)

**Format de sortie** : `data/events.json` : une ligne par chunk, avec le vecteur et les métadonnées associées.

---


## Modèles/libs utilisés

| Rôle | Modèle | Justification |
|------|--------|---------------|
| Embeddings | Qwen3-Embedding-0.6B | Gratuit, open source, multilingue|
| LLM | mistral-small-latest | API gratuite |
| Vectorstore | FAISS IndexFlatIP | compare la requête à tous les vecteurs (exact mais lent sur bcp d'entrées, suffisant pour un POC avec milliers d'événements) |
| Orchestration | LangChain | Standard, facilite la piepline RAG |
| API | FastAPI + Uvicorn | Léger, documentation Swagger automatique, validation des entrées via Pydantic |

---

## Base vectorielle
 
**Construction** (`index_faiss.py`) :
 
Chaque chunk est converti en `Document` LangChain (texte + métadonnées), puis LangChain construit l'index FAISS via `FAISS.from_documents()`.
 
**Persistance** :
 
| Fichier | Contenu |
|---------|---------|
| `index/index.faiss` | Les vecteurs (format binaire FAISS) |
| `index/index.pkl` | Les Documents LangChain avec métadonnées |
 
**Métadonnées conservées** pour chaque chunk : titre, lieu, date de début, URL, description.
 
**Recherche** : LangChain vectorise la question avec le même modèle Qwen3, puis appelle `similarity_search_with_score()` qui retourne les k chunks les plus proches avec leur score de similarité cosinus (entre 0 et 1).
 
---

## Modèle LLM

- *Modèle sélectionné* : mistral-small-latest de Mistral 
- *Pourquoi ce modèle ?* : Gratuit
- *Prompting* : Prompt minimal, persona "d'assistant culturel" :
    - "Tu es un assistant culturel spécialisé dans les événements locaux. Réponds uniquement à partir des événements fournis ci-dessous. Si aucun événement ne correspond, dis-le clairement sans inventer. Réponds en français, de manière concise."
- *Limites du modèle* : limite de tokens car gratuit, pas le modèle le + performant à ce jour


---

## Evaluation du système

- *Jeu de test annoté* :
    ○	15 paires question/réponse couvrant différentes catégories
    ○ Voir `evaluation\event_15.json`.
 
- *Méthode d'annotation* : manuelle, à partir des données réelles d'`events.json`.

- *Métriques d’évaluation* :
    ○ Score de similarité FAISS : cosinus entre vecteur requête et chunks retrouvés. Indique si la recherche vectorielle trouve les bons documents.
    ○ Classification manuelle : correcte / partiellement correcte / incorrecte. Une réponse est correcte si elle contient les mêmes informations que la réponse de référence.
    ○ Taux de couverture : parmi les sources retournées, combien contiennent l'événement attendu ?

- *Résultats obtenus* :
    ○	Analyse quantitative (scores globaux) :
    ○	Analyse qualitative (exemples de bonnes/mauvaises réponses) :

---

## Bilan et perspectives

- *Ce qui fonctionne bien* :
- *Limites du POC* :
    ○ Index statique : il faut relancer manuellement `scrap_clean_embed.py` + `index_faiss.py` quand les données changent
    ○ Pas d'historique de conversation : chaque question est indépendante

- *Améliorations possibles* :
    ○ Déploiement cloud (AWS Lambda ou Azure Container Apps) avec CI/CD GitHub Actions
    ○ Ajouter un filtre par date avant la recherche vectorielle (metadata filtering FAISS)

---

## Annexes (exemples)
●	Extraits du jeu de test annoté
●	Extraits de logs ou exemples de réponse JSON


---

## Variables d'environnement

| Variable | Description | Obligatoire |
|----------|-------------|-------------|
| `MISTRAL_API_KEY` | Clé API Mistral | Oui |