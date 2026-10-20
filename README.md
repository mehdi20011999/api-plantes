# API Plantes : un modèle d'IA mis en production

API FastAPI qui reçoit la photo d'une feuille et renvoie la maladie la plus probable (TensorFlow Lite),
emballée dans Docker (utilisateur non-root, healthcheck) et vérifiée par GitHub Actions à chaque push.

## Lancer
```bash
pip install -r requirements-dev.txt
pytest -v
uvicorn app.main:app --reload    # puis http://localhost:8000/docs
docker build -t api-plantes . && docker run -p 8000:8000 api-plantes
```

## Ce que le projet montre
- Prétraitement identique à l'entraînement, testé
- Modèle chargé une seule fois, runtime TFLite léger, temps d'inférence mesuré
- API robuste : validation des entrées, erreurs claires, limite de taille
- Tests unitaires, tests d'API avec un faux modèle, test de bout en bout
- Docker non-root, healthcheck, intégration continue

> Le modèle fourni dans `models/` est un modèle de démonstration non entraîné, à remplacer par le modèle entraîné.
