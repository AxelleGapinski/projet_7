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

# chargement de l'index FAISS LangChain depuis le disque
vectorstore = FAISS.load_local(
    INDEX_DIR,
    embedder,
    allow_dangerous_deserialization=True
)

# LLM Mistral
llm = ChatMistralAI(
    mistral_api_key=MISTRAL_API_KEY,
    model="mistral-small-latest",
    temperature=0.2
)

# prompt envoyé à Mistral — {context} et {question} sont injectés automatiquement par LangChain
prompt = ChatPromptTemplate.from_template("""Tu es un assistant culturel spécialisé dans les événements locaux.
Réponds uniquement à partir des événements fournis ci-dessous.
Si aucun événement ne correspond, dis-le clairement sans inventer.
Réponds en français, de manière concise.

Événements disponibles :
{context}

Question : {question}

Réponse :""")

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# chaîne RAG LangChain — orchestration automatique FAISS + prompt + Mistral
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
  

def ask(question: str, top_k: int = 5) -> dict:
    """
    Fonction principale appelée par l'API.
    top_k permet de changer le nombre de chunks utilisés dynamiquement.
    """
    r = vectorstore.as_retriever(search_kwargs={"k": top_k})
    c = (
        {"context": r, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = c.invoke(question)

    # pour les sources on fait une recherche séparée
    sources_docs = vectorstore.similarity_search(question, k=top_k)
    sources = [
        {
            "title": doc.metadata.get("title", ""),
            "location": doc.metadata.get("location", ""),
            "date": doc.metadata.get("first_date", ""),
            "url": doc.metadata.get("url", ""),
        }
        for doc in sources_docs
    ]

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