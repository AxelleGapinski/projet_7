import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import json
import csv
from rag_langchain import ask

# chargement jeu de test
with open("evaluation/test_dataset.json", encoding="utf-8") as f:
    test_set = json.load(f)

results = []
notes = []

print("Évaluation du Rag\n")

for item in test_set:
    question = item["question"]
    reponse_attendue = item.get("reponse_attendue", "")
    note_manuelle = item.get("note", None)

    # appel au RAG
    result = ask(question, top_k=10)
    time.sleep(3)
    answer = result["answer"]
    sources = result["sources"]

    # scores FAISS
    scores_faiss = [s["score"] for s in sources if "score" in s]
    score_max = round(max(scores_faiss), 3) if scores_faiss else 0
    score_moyen = round(sum(scores_faiss) / len(scores_faiss), 3) if scores_faiss else 0

    print(f"Score Faiss max : {score_max}")
    print(f"Score faiss moyen: {score_moyen}")    

    #métriques basiques
    nb_sources = len(sources)
    reponse_non_vide = len(answer.strip()) > 10

    # mots clés de la réponse attendue présents dans la réponse générée
    mots_reference = set(reponse_attendue.lower().split()) if reponse_attendue else set()
    mots_reponse = set(answer.lower().split())
    mots_communs = mots_reference & mots_reponse
    couverture = round(len(mots_communs) / len(mots_reference), 2) if mots_reference else 0

    print(f"Q{item['id']} : {question[:60]}...")
    print(f"Sources retrouvées : {nb_sources}")
    print(f"Couverture lexicale : {couverture}")
    print(f"Note manuelle attribuée: {note_manuelle}/5")
    print()

    results.append({
        "id": item["id"],
        "question": question,
        "reponse_generee": answer,
        "nb_sources": nb_sources,
        "couverture_lexicale": couverture,
        "reponse_non_vide": reponse_non_vide,
        "note_manuelle": note_manuelle,
        "score_faiss_max": score_max,
        "score_faiss_moyen": score_moyen,
    })

    if note_manuelle is not None:
        notes.append(note_manuelle)

# résumé global
print("Résumé")
print(f"Questions évaluées : {len(results)}")
if notes:
    print(f"Note moyenne : {round(sum(notes)/len(notes), 2)}/5")
    print(f"Notes 5/5: {sum(1 for n in notes if n == 5)}/{len(notes)}")
    print(f"Notes >= 3/5: {sum(1 for n in notes if n >= 3)}/{len(notes)}")
    print(f"Notes < 3/5 : {sum(1 for n in notes if n < 3)}/{len(notes)}")
print(f"Réponses non vides: {sum(r['reponse_non_vide'] for r in results)}/{len(results)}")
print(f"Sources moyennes: {round(sum(r['nb_sources'] for r in results)/len(results), 1)} par question")
print(f"Score FAISS moyen global: {round(sum(r['score_faiss_moyen'] for r in results)/len(results), 3)}")
print(f"Score FAISS max moyen  : {round(sum(r['score_faiss_max'] for r in results)/len(results), 3)}")

# sauvegarde CSV
os.makedirs("evaluation", exist_ok=True)
with open("evaluation/results.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("\nRésultats sauvegardés dans evaluation/results.csv")