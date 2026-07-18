"""
Chargement du modèle (une seule fois, au démarrage) et conversion d'une
requête utilisateur (RentalFeatures) en vecteur de features aligné sur celui
utilisé à l'entraînement.
"""

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator

from senegal_rental_price.features.build_features import (
    align_features,
    encode_categorical,
    parse_equipements,
)
from senegal_rental_price.models.predict import load_model
from senegal_rental_price.utils.logger import get_logger

from api.schemas import RentalFeatures

logger = get_logger(__name__)

# Surchargeable via variable d'environnement (utile en conteneur Docker, où
# le dossier des modèles peut être monté à un autre chemin qu'en local).
MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))
DEFAULT_MODEL_NAME = "random_forest"  # le meilleur des 3 modèles comparés (cf. rapport)


@lru_cache(maxsize=1)
def get_model() -> BaseEstimator:
    """Charge le modèle une seule fois au démarrage (mis en cache pour tout le cycle de vie de l'API)."""
    model_path = MODELS_DIR / f"{DEFAULT_MODEL_NAME}.pkl"
    return load_model(model_path)


@lru_cache(maxsize=1)
def get_feature_columns() -> list[str]:
    """Charge la liste des colonnes de features vues à l'entraînement."""
    features_path = MODELS_DIR / f"{DEFAULT_MODEL_NAME}_features.json"
    with open(features_path, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def get_model_metadata() -> dict[str, Any]:
    """Charge les métadonnées du modèle (version, date d'entraînement, métriques)."""
    metadata_path = MODELS_DIR / f"{DEFAULT_MODEL_NAME}_metadata.json"
    with open(metadata_path, encoding="utf-8") as f:
        return json.load(f)


def build_input_dataframe(features: RentalFeatures) -> pd.DataFrame:
    """
    Convertit une requête RentalFeatures en DataFrame prêt pour le modèle :
    même parsing d'équipements, même encodage catégoriel, puis alignement
    strict sur les colonnes vues à l'entraînement.

    Les colonnes `*_imputee`/`surface_estimee` (artefacts du nettoyage des
    données d'entraînement) sont fixées à False : une requête API fournit
    toujours des valeurs réelles, jamais imputées/estimées.
    """
    raw = pd.DataFrame(
        [
            {
                "ville": features.ville.value,
                "type_bien": features.type_bien.value,
                "surface_m2": features.surface_m2,
                "surface_estimee": False,
                "nb_pieces": features.nb_pieces,
                "nb_pieces_imputee": False,
                "nb_chambres": features.nb_chambres,
                "nb_chambres_imputee": False,
                "meuble": features.meuble,
                "equipements": "|".join(features.equipements),
            }
        ]
    )

    df = parse_equipements(raw)
    df = encode_categorical(df)

    feature_columns = get_feature_columns()
    return align_features(df, feature_columns)
