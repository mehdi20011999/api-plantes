"""Chargement du modèle TFLite et prédiction sur une image."""
from pathlib import Path

import numpy as np
from PIL import Image
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

IMG_SIZE = 224


def preprocess(image: Image.Image) -> np.ndarray:
    """Prépare une image comme à l'entraînement : RGB, 224x224, valeurs dans [-1, 1], axe batch."""
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    x = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return x[np.newaxis, ...]


def top_k(probs: np.ndarray, labels: list[str], k: int = 3) -> list[tuple[str, float]]:
    """Renvoie les k classes les plus probables, de la plus probable à la moins probable."""
    probs = np.asarray(probs).ravel()
    idx = np.argsort(probs)[::-1][:k]
    return [(labels[i], round(float(probs[i]), 4)) for i in idx]


class Classifier:
    def __init__(self, model_path, labels_path):
        self.interpreter = Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]
        self.labels = [l.strip() for l in Path(labels_path).read_text(encoding="utf-8").splitlines() if l.strip()]

    def predict(self, image: Image.Image, k: int = 3) -> list[tuple[str, float]]:
        self.interpreter.set_tensor(self.input["index"], preprocess(image))
        self.interpreter.invoke()
        probs = self.interpreter.get_tensor(self.output["index"])[0]
        return top_k(probs, self.labels, k)
