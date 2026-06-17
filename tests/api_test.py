import requests

BASE_URL = "http://localhost:8000"


def test_root():
    """Vérifie que l'API répond"""
    r = requests.get(f"{BASE_URL}/")
    assert r.status_code == 200
    print("GET /  = ", r.json())


def test_ask_normal():
    """Question normale qui doit retourner une réponse + des sources"""
    r = requests.post(f"{BASE_URL}/ask", json={"question": "ya-t-il des concerts ce mois-ci ?"})
    assert r.status_code == 200
    data = r.json()
    assert "answer" in data
    assert "sources" in data
    print("POST /ask= ", data["answer"][:100], "...")


def test_ask_empty():
    """Question vide: doit retourner une erreur 400"""
    r = requests.post(f"{BASE_URL}/ask", json={"question": ""})
    assert r.status_code == 400
    print("POST /ask (vide)= erreur 400")

if __name__ == "__main__":
    test_root()
    test_ask_normal()
    test_ask_empty()
    print("\nTests ok")