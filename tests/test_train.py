"""
Tests unitaires pour senegal_rental_price.models.train.

On ne teste pas `main()` directement (décoré par @hydra.main, nécessite un
vrai contexte Hydra) mais les fonctions pures qu'il utilise, qui contiennent
toute la logique métier testable.
"""

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from senegal_rental_price.models.train import cross_validate_model, evaluate, get_model


class TestGetModel:
    def test_returns_ridge_with_correct_params(self) -> None:
        cfg = OmegaConf.create({"model": {"name": "ridge", "params": {"alpha": 2.0}}})
        model = get_model(cfg)
        assert isinstance(model, Ridge)
        assert model.alpha == 2.0

    def test_returns_random_forest_with_correct_params(self) -> None:
        cfg = OmegaConf.create(
            {"model": {"name": "random_forest", "params": {"n_estimators": 50, "max_depth": 5}}}
        )
        model = get_model(cfg)
        assert isinstance(model, RandomForestRegressor)
        assert model.n_estimators == 50
        assert model.max_depth == 5

    def test_returns_xgboost_with_correct_params(self) -> None:
        pytest.importorskip("xgboost")
        cfg = OmegaConf.create(
            {"model": {"name": "xgboost", "params": {"n_estimators": 100, "max_depth": 3}}}
        )
        model = get_model(cfg)
        assert model.n_estimators == 100

    def test_raises_for_unknown_model(self) -> None:
        cfg = OmegaConf.create({"model": {"name": "modele_inconnu", "params": {}}})
        with pytest.raises(ValueError):
            get_model(cfg)


class TestEvaluate:
    def test_returns_expected_metrics(self) -> None:
        y_true = pd.Series([100_000, 200_000, 300_000])
        y_pred = [110_000, 190_000, 310_000]

        metrics = evaluate(y_true, y_pred)

        assert set(metrics.keys()) == {"mae", "rmse", "r2"}
        assert metrics["mae"] > 0
        assert metrics["rmse"] > 0

    def test_perfect_prediction_gives_zero_error(self) -> None:
        y_true = pd.Series([100_000, 200_000, 300_000])
        y_pred = [100_000, 200_000, 300_000]

        metrics = evaluate(y_true, y_pred)

        assert metrics["mae"] == 0
        assert metrics["rmse"] == 0
        assert metrics["r2"] == 1.0


class TestCrossValidateModel:
    @pytest.fixture
    def toy_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """Jeu de données synthétique suffisant pour tester la mécanique de la CV (pas la qualité du fit)."""
        rng = np.random.default_rng(42)
        n = 50
        X = pd.DataFrame(
            {
                "surface_m2": rng.uniform(20, 300, n),
                "nb_pieces": rng.integers(1, 8, n),
            }
        )
        y = pd.Series(50_000 + X["surface_m2"] * 5_000 + rng.normal(0, 50_000, n))
        return X, y

    def test_returns_expected_keys(self, toy_data: tuple[pd.DataFrame, pd.Series]) -> None:
        X, y = toy_data
        result = cross_validate_model(Ridge(), X, y, n_splits=5, seed=42)
        expected_keys = {
            "cv_mae_mean",
            "cv_mae_std",
            "cv_rmse_mean",
            "cv_rmse_std",
            "cv_r2_mean",
            "cv_r2_std",
        }
        assert set(result.keys()) == expected_keys

    def test_metrics_are_non_negative_where_expected(
        self, toy_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = toy_data
        result = cross_validate_model(Ridge(), X, y, n_splits=5, seed=42)
        assert result["cv_mae_mean"] >= 0
        assert result["cv_rmse_mean"] >= 0
        assert result["cv_mae_std"] >= 0

    def test_is_deterministic_with_fixed_seed(
        self, toy_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = toy_data
        result_1 = cross_validate_model(Ridge(), X, y, n_splits=5, seed=42)
        result_2 = cross_validate_model(Ridge(), X, y, n_splits=5, seed=42)
        assert result_1 == result_2

    def test_does_not_mutate_original_model(self, toy_data: tuple[pd.DataFrame, pd.Series]) -> None:
        X, y = toy_data
        model = Ridge()
        cross_validate_model(model, X, y, n_splits=5, seed=42)
        # Le modèle passé en argument ne doit jamais être fit lui-même (on clone
        # à chaque pli) : il ne doit donc pas avoir d'attribut "coef_" après coup.
        assert not hasattr(model, "coef_")
