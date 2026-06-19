import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
INDEX_DIR = "index"

# chargement modèle d'embeddings QWEN
embedder = HuggingFaceEmbeddings(
    model_name="Qwen/Qwen3-Embedding-0.6B",
    model_kwargs={"trust_remote_code": True},
    encode_kwargs={"normalize_embeddings": True}
)

#chargement de l'index FAISS LangChain
vectorstore = FAISS.load_local(
    INDEX_DIR,
    embedder,
    allow_dangerous_deserialization=True
)

#LLM Mistral
llm = ChatMistralAI(
    mistral_api_key=MISTRAL_API_KEY,
    model="mistral-small-latest",
    temperature=0.2
)

# prompt envoyé à Mistral
prompt = ChatPromptTemplate.from_template("""
                                          
# RÔLE 
Tu es un assistant culturel spécialisé dans les événements locaux. Tu dois répondre aux questions des utilisateurs au sujet d'évènements ayant lieu dans certains lieux, à certaines dates ou correspondant à des thèmes spécifiques.
                                          
## REGLES
Réponds à la question uniquement à partir des événements fournis ci-dessous. 
Réponds en français, de façon entousiasthe et fournis toutes les informations qui sont en ta possession. Ne tronques pas tes phrases et fournis des informations complètes SI tu les as à disposition. Si tu n'as pas l'heure de l'évènement par exemple, ne l'invente pas. 
Sois exhaustif : si plusieurs événements correspondent à la question, liste-les tous.
Si aucun événement ne correspond à la question, dis-le clairement sans inventer d'information. Tu peux suggérer des évènements similaires si aucun ne correspond exactement à la question.
S'il y a des liens web disponibles pour les événements, fournis-les dans ta réponse.
                                          
### Exemples de questions et de réponses attendues :
 - Question 1 : "y a t-il des évènements en rapport avec les chiens en juin ?"
 - Réponse 1 :  "Le 7 juin à Saint-Médard-en-Jalles, il y a une compétition de hoopers organisée par le club Tactichien. Il y aura une buvette et des stands. Les 13 et 14 juin, un concours d'obéissance est également prévu au même endroit !"

 - Question 2 : "y a t-il des évènements en rapport avec les chiens en mai ?"
 - Réponse 2 :  "Il n'y a pas d'évènements en rapport avec les chiens en mai. En revanche, le 7 juin à Saint-Médard-en-Jalles, il y a une compétition de hoopers organisée par le club Tactichien. Il y aura une buvette et des stands. Les 13 et 14 juin, un concours d'obéissance est également prévu au même endroit !"
                                          
 - Question 3 : "Où est la meilleure pizzeria de Bordeaux ?"
 - Réponse 3 :  "Je suis un chatbot spécialisé dans les évènements culturels, je n'ai pas réponse à cette question."
                                                                                     
# CONTEXTE
                                          
### Événements disponibles :
{context}

### Question : {question}

### Réponse :""")

retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# rag (LangChain) = orchestration automatique faissd + prompt + Mistral
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
  

def ask(question: str, top_k: int = 10) -> dict:
    """
    Fonction principale appelée par l'API
    top_k permet de changer le nombre de chunks utilisés
    """
    r = vectorstore.as_retriever(search_kwargs={"k": top_k})
    c = (
        {"context": r, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = c.invoke(question)

    #recherche avec scores pour les sources
    sources_docs_scores = vectorstore.similarity_search_with_score(question, k=top_k)

    sources_vues = set()
    sources = []
    for doc, score in sources_docs_scores:
        titre = doc.metadata.get("title", "")
        if titre not in sources_vues:
            sources_vues.add(titre)
            sources.append({
                "title": titre,
                "location": doc.metadata.get("location", ""),
                "date": doc.metadata.get("first_date", ""),
                "url": doc.metadata.get("url", ""),
                "score": round(float(score), 3),
            })

    return {
        "question": question,
        "answer": answer,
        "sources": sources
    }


if __name__ == "__main__":
    result = ask("est-ce qu'il y a des concerts ce mois-ci ?")
    print("Réponse :", result["answer"])
    print("\nSources :")
    for s in result["sources"]:
        print(f"  - {s['title']} | {s['location']}")