"""Étapes 1 à 4 : le prétraitement, le top-k et le vrai modèle."""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.model import Classifier, preprocess, top_k

ROOT = Path(__file__).resolve().parent.parent


# --- Étape 1 : preprocess -------------------------------------------------------------
def test_preprocess_forme_et_type():
    x = preprocess(Image.new("RGB", (640, 480)))
    assert x.shape == (1, 224, 224, 3), "attendu : un batch d'une image 224x224 en RGB"
    assert x.dtype == np.float32, "le modèle attend des float32"


def test_preprocess_convertit_en_rgb():
    x = preprocess(Image.new("L", (100, 100)))          # image en niveaux de gris
    assert x.shape == (1, 224, 224, 3), "convertis l'image en RGB avant tout"


def test_preprocess_normalise_entre_moins_un_et_un():
    noir = preprocess(Image.new("RGB", (50, 50), (0, 0, 0)))
    blanc = preprocess(Image.new("RGB", (50, 50), (255, 255, 255)))
    assert np.allclose(noir, -1.0), "un pixel noir (0) doit valoir -1"
    assert np.allclose(blanc, 1.0), "un pixel blanc (255) doit valoir 1"


# --- Étape 2 : top_k ------------------------------------------------------------------
def test_top_k_trie_du_plus_probable_au_moins_probable():
    res = top_k(np.array([0.1, 0.7, 0.2]), ["a", "b", "c"], k=3)
    assert [c for c, _ in res] == ["b", "c", "a"]


def test_top_k_respecte_k_et_renvoie_des_float():
    res = top_k(np.array([[0.5, 0.3, 0.15, 0.05]]), ["a", "b", "c", "d"], k=2)
    assert len(res) == 2
    assert isinstance(res[0][1], float), "convertis les scores numpy en float Python (sinon le JSON échoue)"


# --- Étapes 3 et 4 : Classifier avec le vrai fichier .tflite --------------------------
@pytest.fixture(scope="module")
def clf():
    model = ROOT / "models" / "modele.tflite"
    if not model.exists():
        pytest.skip("models/modele.tflite absent : lance scripts/creer_modele_demo.py ou ajoute ton modèle")
    return Classifier(model, ROOT / "models" / "labels.txt")


def test_classifier_charge_les_labels(clf):
    assert len(clf.labels) >= 2
    assert all(label == label.strip() and label for label in clf.labels), "retire les espaces et lignes vides"


def test_classifier_predit_un_top3_valide(clf):
    res = clf.predict(Image.new("RGB", (300, 300), (30, 120, 40)))
    assert 1 <= len(res) <= 3
    classes = [c for c, _ in res]
    assert all(c in clf.labels for c in classes)
    scores = [p for _, p in res]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= p <= 1.0 for p in scores)


# --- De bout en bout : l'API avec le vrai modèle -----------------------------------------
def test_api_avec_le_vrai_modele(clf, png):
    from fastapi.testclient import TestClient
    from app.main import app
    r = TestClient(app).post("/predict", files={"file": ("feuille.png", png, "image/png")})
    assert r.status_code == 200, r.text
    assert r.json()["classe"] in clf.labels
