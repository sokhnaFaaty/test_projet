"""
API FastAPI exposant le modèle de prédiction du prix de location au Sénégal.

Lancement :
    uvicorn senegal_rental_price.api.main:app --reload

Documentation interactive : http://localhost:8000/docs
"""

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from senegal_rental_price.models.predict import predict
from senegal_rental_price.utils.logger import get_logger

from api.dependencies import (
    build_input_dataframe,
    get_model,
    get_model_metadata,
)
from api.schemas import HealthResponse, ModelInfo, PredictionResponse, RentalFeatures

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Charge le modèle une seule fois au démarrage (pas à chaque requête)."""
    try:
        get_model()
        get_model_metadata()
        logger.info("Modèle chargé avec succès au démarrage de l'API.")
    except FileNotFoundError as exc:
        logger.error("Impossible de charger le modèle au démarrage : %s", exc)
        raise
    yield
    logger.info("Arrêt de l'API.")


app = FastAPI(
    title="API de prédiction du prix des locations au Sénégal",
    description=(
        "Prédit le prix mensuel de location (FCFA) d'un bien immobilier "
        "(appartement ou maison) au Sénégal, à partir de ses caractéristiques."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS : sans ça, le navigateur bloque silencieusement les appels venant du
# frontend vers cette API si elles ne sont pas sur la même origine. La liste
# d'origines autorisées est injectée par variable d'environnement (CORS_ORIGINS,
# séparées par des virgules) plutôt que codée en dur, pour permettre de changer
# de configuration entre dev local et conteneurs Docker sans toucher au code.
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health() -> HealthResponse:
    """Vérifie que le service est opérationnel."""
    return HealthResponse(status="ok")


@app.get("/model/info", response_model=ModelInfo, tags=["Monitoring"])
def model_info() -> ModelInfo:
    """Retourne les métadonnées du modèle actuellement chargé (version, date, métriques)."""
    try:
        metadata = get_model_metadata()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Métadonnées du modèle indisponibles.") from exc
    return ModelInfo(**metadata)


@app.post("/predict", response_model=PredictionResponse, tags=["Prédiction"])
def predict_price(features: RentalFeatures) -> PredictionResponse:
    """Prédit le prix de location mensuel (FCFA) à partir des caractéristiques d'un bien."""
    try:
        model = get_model()
        metadata = get_model_metadata()
        X = build_input_dataframe(features)
        prediction = predict(model, X)[0]
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Modèle indisponible.") from exc

    return PredictionResponse(
        prix_loyer_mensuel_estime=round(prediction, -2),  # arrondi à la centaine près
        model_version=metadata["version"],
    )
