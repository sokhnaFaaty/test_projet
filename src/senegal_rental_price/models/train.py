"""
Entraînement d'un modèle de prédiction du prix de location, avec configuration
Hydra (aucun hyperparamètre codé en dur) et tracking MLflow.

Usage :
    python -m senegal_rental_price.models.train
    python -m senegal_rental_price.models.train model=xgboost model.params.max_depth=8
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import hydra
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

from senegal_rental_price.features.build_features import build_feature_matrix
from senegal_rental_price.utils.logger import get_logger

logger = get_logger(__name__)

MLFLOW_EXPERIMENT_NAME = "senegal-rental-price"


def get_model(cfg: DictConfig) -> BaseEstimator:
    """Instancie le modèle scikit-learn/xgboost correspondant à la config Hydra."""
    name = cfg.model.name
    params: dict[str, Any] = dict(cfg.model.params)

    if name == "ridge":
        return Ridge(**params)
    if name == "random_forest":
        return RandomForestRegressor(**params)
    if name == "xgboost":
        from xgboost import XGBRegressor  # import tardif : dépendance optionnelle

        return XGBRegressor(**params)

    raise ValueError(f"Modèle inconnu : '{name}'. Attendu : ridge, random_forest, xgboost.")


def evaluate(y_true: pd.Series, y_pred: Any) -> dict[str, float]:
    """Calcule les métriques de régression standard, sur l'échelle réelle des prix (FCFA)."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def cross_validate_model(
    model: BaseEstimator, X: pd.DataFrame, y: pd.Series, n_splits: int = 5, seed: int = 42
) -> dict[str, float]:
    """
    Validation croisée à `n_splits` plis, plus robuste qu'un split train/test
    unique sur un petit échantillon (166 lignes) où un seul découpage peut
    fortement biaiser les métriques selon les lignes qui tombent côté test.

    Le modèle est réentraîné (via `clone`) sur chaque pli, sur log1p(prix) —
    cohérent avec l'entraînement final — puis évalué sur l'échelle réelle (FCFA).
    Retourne la moyenne et l'écart-type de chaque métrique sur les `n_splits` plis.
    """
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_metrics: dict[str, list[float]] = {"mae": [], "rmse": [], "r2": []}

    for train_idx, test_idx in kfold.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        fold_model = clone(model)
        fold_model.fit(X_train, np.log1p(y_train))
        y_pred = np.expm1(fold_model.predict(X_test))

        fold_result = evaluate(y_test, y_pred)
        for key, value in fold_result.items():
            fold_metrics[key].append(value)

    summary: dict[str, float] = {}
    for key, values in fold_metrics.items():
        summary[f"cv_{key}_mean"] = float(np.mean(values))
        summary[f"cv_{key}_std"] = float(np.std(values))
    return summary


@hydra.main(config_path="../../../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logger.info("Configuration utilisée :\n%s", OmegaConf.to_yaml(cfg))

    df = pd.read_csv(cfg.data.processed_path, sep=";", encoding="utf-8-sig")
    logger.info("Données chargées : %d lignes", len(df))

    X, y = build_feature_matrix(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.data.test_size, random_state=cfg.seed
    )

    # Entraînement sur log(1 + prix) plutôt que le prix brut : le prix va de
    # 20 000 à 15 000 000 FCFA (plusieurs ordres de grandeur, cf. les annonces
    # marquées `prix_atypique`), ce qui rend l'échelle brute très instable pour
    # un modèle de régression sur un échantillon encore petit. Le log stabilise
    # la variance sans supprimer aucune donnée (décision documentée dans le
    # rapport). Les métriques restent calculées sur l'échelle réelle (FCFA) via
    # np.expm1, pour rester interprétables.
    y_train_log = np.log1p(y_train)

    model = get_model(cfg)

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=cfg.model.name):
        mlflow.log_params(dict(cfg.model.params))
        mlflow.log_param("test_size", cfg.data.test_size)
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("target_transform", "log1p")

        # Validation croisée (5-fold) : estimation plus robuste que le seul
        # split train/test ci-dessous, qui peut être instable sur un aussi
        # petit échantillon. Les deux sont conservées et documentées dans
        # le rapport : la CV pour la fiabilité de l'estimation de performance,
        # le split unique pour le modèle final réellement sauvegardé/déployé.
        cv_metrics = cross_validate_model(clone(model), X, y, n_splits=5, seed=cfg.seed)
        mlflow.log_metrics(cv_metrics)
        logger.info(
            "Validation croisée (5-fold) [%s] -> MAE=%.0f±%.0f RMSE=%.0f±%.0f R2=%.3f±%.3f",
            cfg.model.name,
            cv_metrics["cv_mae_mean"],
            cv_metrics["cv_mae_std"],
            cv_metrics["cv_rmse_mean"],
            cv_metrics["cv_rmse_std"],
            cv_metrics["cv_r2_mean"],
            cv_metrics["cv_r2_std"],
        )

        model.fit(X_train, y_train_log)
        y_pred_log = model.predict(X_test)
        y_pred = np.expm1(y_pred_log)  # retour à l'échelle réelle (FCFA)

        metrics = evaluate(y_test, y_pred)
        mlflow.log_metrics(metrics)

        # XGBoost ne peut pas être sérialisé avec le flavor "sklearn" de MLflow
        # (bloqué par un contrôle de sécurité de skops sur les types non natifs) ;
        # on utilise le flavor dédié pour ce modèle.
        if cfg.model.name == "xgboost":
            mlflow.xgboost.log_model(model, artifact_path="model")
        else:
            mlflow.sklearn.log_model(model, artifact_path="model")

        logger.info(
            "Résultats [%s] -> MAE=%.0f RMSE=%.0f R2=%.3f",
            cfg.model.name,
            metrics["mae"],
            metrics["rmse"],
            metrics["r2"],
        )

        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model_path = output_dir / f"{cfg.model.name}.pkl"
        joblib.dump(model, model_path)
        logger.info("Modèle sauvegardé dans %s", model_path)

        # Colonnes de features vues à l'entraînement : indispensables pour que
        # l'API reconstruise un vecteur de features de forme identique lors de
        # l'inférence (cf. features.build_features.align_features).
        features_path = output_dir / f"{cfg.model.name}_features.json"
        with open(features_path, "w", encoding="utf-8") as f:
            json.dump(list(X.columns), f, ensure_ascii=False, indent=2)
        logger.info("Colonnes de features sauvegardées dans %s", features_path)

        # Métadonnées exposées plus tard par l'endpoint GET /model/info de l'API.
        metadata = {
            "name": cfg.model.name,
            "version": "1.0.0",
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "metrics": {**metrics, **cv_metrics},
        }
        metadata_path = output_dir / f"{cfg.model.name}_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        logger.info("Métadonnées du modèle sauvegardées dans %s", metadata_path)


if __name__ == "__main__":
    main()