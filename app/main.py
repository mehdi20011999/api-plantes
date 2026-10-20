"""API de classification d'images de plantes."""
import io
import os
import time
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from app.model import Classifier

MODEL_PATH = os.getenv("MODEL_PATH", "models/modele.tflite")
LABELS_PATH = os.getenv("LABELS_PATH", "models/labels.txt")
MAX_BYTES = 5 * 1024 * 1024
TYPES_ACCEPTES = {"image/jpeg", "image/png"}

app = FastAPI(title="API Plantes", version="1.0.0",
              description="Envoie une photo de feuille, l'API renvoie la maladie la plus probable.")


@lru_cache
def get_classifier() -> Classifier:
    """Chargé une seule fois, au premier appel, puis réutilisé."""
    return Classifier(MODEL_PATH, LABELS_PATH)


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post("/predict")
async def predict(file: UploadFile, clf: Classifier = Depends(get_classifier)):
    if file.content_type not in TYPES_ACCEPTES:
        raise HTTPException(400, "Envoyez une image JPEG ou PNG")
    data = await file.read()
    if not data:
        raise HTTPException(400, "Fichier vide")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image trop lourde (5 Mo maximum)")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "Image illisible")
    debut = time.perf_counter()
    top3 = clf.predict(image)
    duree = round((time.perf_counter() - debut) * 1000, 1)
    return {"classe": top3[0][0], "confiance": top3[0][1],
            "top3": [{"classe": c, "confiance": p} for c, p in top3], "temps_ms": duree}
