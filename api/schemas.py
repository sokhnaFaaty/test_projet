"""Schémas Pydantic pour la validation des requêtes/réponses de l'API."""

from enum import Enum

from pydantic import BaseModel, Field


class Ville(str, Enum):
    dakar = "Dakar"
    thies = "Thiès"
    saint_louis = "Saint-Louis"
    mbour = "Mbour"
    saly = "Saly"
    ziguinchor = "Ziguinchor"


class TypeBien(str, Enum):
    appartement = "appartement"
    maison = "maison"


class RentalFeatures(BaseModel):
    """Caractéristiques d'un bien immobilier, fournies par l'utilisateur pour obtenir une prédiction."""

    ville: Ville = Field(..., description="Ville ou région du bien")
    type_bien: TypeBien = Field(..., description="Type de bien : appartement ou maison")
    surface_m2: float = Field(..., gt=0, le=2000, description="Surface habitable en m²")
    nb_pieces: int = Field(..., ge=1, le=20, description="Nombre de pièces")
    nb_chambres: int = Field(..., ge=0, le=15, description="Nombre de chambres")
    meuble: bool = Field(..., description="Le bien est-il meublé ?")
    equipements: list[str] = Field(
        default_factory=list,
        description="Équipements présents, ex. ['piscine', 'parking', 'gardiennage']",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "ville": "Dakar",
                    "type_bien": "appartement",
                    "surface_m2": 80,
                    "nb_pieces": 3,
                    "nb_chambres": 2,
                    "meuble": True,
                    "equipements": ["piscine", "parking"],
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    prix_loyer_mensuel_estime: float = Field(..., description="Prix mensuel estimé, en FCFA")
    devise: str = "FCFA"
    model_version: str = Field(..., description="Version du modèle utilisé pour la prédiction")


class ModelInfo(BaseModel):
    name: str
    version: str
    trained_at: str
    metrics: dict[str, float]


class HealthResponse(BaseModel):
    status: str = "ok"
