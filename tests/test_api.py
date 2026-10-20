"""Étape 5 : l'API. Le vrai modèle est remplacé par un faux : ces tests vérifient l'API seule."""
import pytest
from fastapi.testclient import TestClient

from app.main import app, get_classifier


class FauxClassifier:
    labels = ["sain", "mildiou", "rouille"]

    def predict(self, image, k=3):
        return [("mildiou", 0.91), ("rouille", 0.06), ("sain", 0.03)]


@pytest.fixture
def client():
    app.dependency_overrides[get_classifier] = FauxClassifier
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_png(client, png):
    r = client.post("/predict", files={"file": ("feuille.png", png, "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["classe"] == "mildiou"
    assert body["confiance"] == 0.91
    assert len(body["top3"]) == 3 and body["top3"][0] == {"classe": "mildiou", "confiance": 0.91}
    assert body["temps_ms"] >= 0


def test_predict_jpeg(client, jpeg):
    r = client.post("/predict", files={"file": ("feuille.jpg", jpeg, "image/jpeg")})
    assert r.status_code == 200, r.text


def test_refuse_un_fichier_texte(client):
    r = client.post("/predict", files={"file": ("notes.txt", b"bonjour", "text/plain")})
    assert r.status_code == 400


def test_refuse_un_fichier_vide(client):
    r = client.post("/predict", files={"file": ("vide.png", b"", "image/png")})
    assert r.status_code == 400


def test_refuse_une_fausse_image(client):
    r = client.post("/predict", files={"file": ("faux.png", b"ceci n'est pas une image", "image/png")})
    assert r.status_code == 400


def test_refuse_une_image_trop_lourde(client):
    r = client.post("/predict", files={"file": ("lourde.png", b"0" * (5 * 1024 * 1024 + 1), "image/png")})
    assert r.status_code == 413
