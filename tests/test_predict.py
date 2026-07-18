"""Tests unitaires pour senegal_rental_price.models.predict."""

from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from senegal_rental_price.models.predict import load_model, predict


@pytest.fixture
def fake_model_path(tmp_path: Path) -> Path:
    """Entraîne un modèle factice minimal et le sauvegarde, pour tester le chargement."""
    X = pd.DataFrame({"surface_m2": [50, 80, 120], "nb_pieces": [2, 3, 4]})
    y = pd.Series([300_000, 500_000, 700_000])

    model = LinearRegression()
    model.fit(X, y)

    model_path = tmp_path / "model.pkl"
    joblib.dump(model, model_path)
    return model_path


class TestLoadModel:
    def test_loads_existing_model(self, fake_model_path: Path) -> None:
        model = load_model(fake_model_path)
        assert isinstance(model, LinearRegression)

    def test_raises_if_model_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_model(tmp_path / "inexistant.pkl")


class TestPredict:
    def test_predict_returns_list_of_floats(self, fake_model_path: Path) -> None:
        model = load_model(fake_model_path)
        X = pd.DataFrame({"surface_m2": [60], "nb_pieces": [2]})

        result = predict(model, X)

        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)

    def test_predict_matches_input_length(self, fake_model_path: Path) -> None:
        model = load_model(fake_model_path)
        X = pd.DataFrame({"surface_m2": [60, 90], "nb_pieces": [2, 3]})

        result = predict(model, X)

        assert len(result) == 2
