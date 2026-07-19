"""
Chargement d'un modèle entraîné et prédiction sur de nouvelles données.

Réutilisé par l'API FastAPI (api/main.py) pour exposer le modèle en production.

Important : les modèles entraînés par `models/train.py` apprennent sur
log1p(prix_loyer_mensuel) plutôt que le prix brut (cf. train.py pour la
justification). `predict()` applique donc automatiquement la retransformation
inverse (expm1) pour renvoyer un prix directement interprétable en FCFA.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from senegal_rental_price.utils.logger import get_logger

logger = get_logger(__name__)


def load_model(model_path: Path) -> BaseEstimator:
    """Charge un modèle scikit-learn/xgboost sérialisé avec joblib."""
    if not model_path.exists():
        raise FileNotFoundError(f"Modèle introuvable : {model_path}")
    logger.info("Chargement du modèle depuis %s", model_path)
    return joblib.load(model_path)


def predict(model: BaseEstimator, X: pd.DataFrame) -> list[float]:
    """
    Prédit le prix de location (en FCFA) pour un ou plusieurs biens déjà
    transformés en features. Applique automatiquement expm1 pour retransformer
    la prédiction du modèle (faite en échelle log1p) vers l'échelle réelle.
    """
    predictions_log = model.predict(X)
    predictions = np.expm1(predictions_log)
    return [float(p) for p in predictions]
