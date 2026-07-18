"""Tests unitaires pour l'API FastAPI, avec un modèle factice pour l'isolation."""

import json
from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LinearRegression

import api.dependencies as deps
from api.main import app


@pytest.fixture
def fake_models_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """
    Crée un modèle factice + ses métadonnées dans un dossier temporaire, et
    redirige l'API vers ce dossier (isolation complète des tests, aucune
    dépendance à un vrai modèle entraîné sur disque).
    """
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    feature_columns = [
        "surface_m2",
        "nb_pieces",
        "nb_chambres",
        "meuble",
        "equip_piscine",
        "equip_parking",
        "ville_Thiès",
        "type_bien_maison",
    ]
    X = pd.DataFrame([[80, 3, 2, True, False, True, False, False]], columns=feature_columns)
    y = pd.Series([13.5])  # échelle log1p, cohérent avec train.py
    model = LinearRegression().fit(X, y)
    joblib.dump(model, models_dir / "random_forest.pkl")

    with open(models_dir / "random_forest_features.json", "w", encoding="utf-8") as f:
        json.dump(feature_columns, f)

    metadata = {
        "name": "random_forest",
        "version": "1.0.0",
        "trained_at": "2026-07-17T10:00:00+00:00",
        "metrics": {"mae": 300000.0, "rmse": 400000.0, "r2": 0.35},
    }
    with open(models_dir / "random_forest_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    monkeypatch.setattr(deps, "MODELS_DIR", models_dir)
    deps.get_model.cache_clear()
    deps.get_feature_columns.cache_clear()
    deps.get_model_metadata.cache_clear()

    return models_dir


@pytest.fixture
def client(fake_models_dir: Path) -> TestClient:
    return TestClient(app)


VALID_PAYLOAD = {
    "ville": "Dakar",
    "type_bien": "appartement",
    "surface_m2": 80,
    "nb_pieces": 3,
    "nb_chambres": 2,
    "meuble": True,
    "equipements": ["piscine", "parking"],
}


class TestHealth:
    def test_health_returns_ok(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestModelInfo:
    def test_model_info_returns_metadata(self, client: TestClient) -> None:
        response = client.get("/model/info")
        assert response.status_code == 200
        body = response.json()
        assert body["name"] == "random_forest"
        assert body["version"] == "1.0.0"
        assert "metrics" in body


class TestPredict:
    def test_predict_with_valid_payload_returns_200(self, client: TestClient) -> None:
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert response.status_code == 200

    def test_predict_response_has_expected_fields(self, client: TestClient) -> None:
        response = client.post("/predict", json=VALID_PAYLOAD)
        body = response.json()
        assert "prix_loyer_mensuel_estime" in body
        assert body["devise"] == "FCFA"
        assert body["model_version"] == "1.0.0"

    def test_predict_price_is_positive(self, client: TestClient) -> None:
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert response.json()["prix_loyer_mensuel_estime"] > 0

    def test_predict_rejects_negative_surface(self, client: TestClient) -> None:
        payload = {**VALID_PAYLOAD, "surface_m2": -10}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_rejects_invalid_ville(self, client: TestClient) -> None:
        payload = {**VALID_PAYLOAD, "ville": "VilleInexistante"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_rejects_missing_required_field(self, client: TestClient) -> None:
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "surface_m2"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_rejects_negative_nb_pieces(self, client: TestClient) -> None:
        payload = {**VALID_PAYLOAD, "nb_pieces": 0}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_accepts_empty_equipements(self, client: TestClient) -> None:
        payload = {**VALID_PAYLOAD, "equipements": []}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

    def test_predict_rejects_surface_above_max(self, client: TestClient) -> None:
        payload = {**VALID_PAYLOAD, "surface_m2": 5000}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
