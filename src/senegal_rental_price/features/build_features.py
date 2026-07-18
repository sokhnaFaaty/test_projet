"""
Fonctions de transformation des données nettoyées en features exploitables
par un modèle de machine learning.

Différence avec `data/preprocessing.py` : ce module ne corrige plus la qualité
des données (déjà fait), il les TRANSFORME pour l'apprentissage (encodage,
parsing, sélection de colonnes).
"""

from typing import Sequence

import pandas as pd

from senegal_rental_price.utils.logger import get_logger

logger = get_logger(__name__)

# Doit correspondre aux clés utilisées lors de l'extraction dans le scraper
# (EQUIPMENT_KEYWORDS de scripts/scrape_neobien.py), pour rester cohérent
# de bout en bout entre la collecte et l'entraînement.
EQUIPEMENTS_CONNUS = [
    "piscine",
    "climatisation",
    "gardiennage",
    "parking",
    "meuble",
    "wifi",
    "jardin",
    "terrasse",
    "salle_de_sport",
]

# Colonnes purement informatives/traçabilité, jamais utilisées comme features.
# `prix_atypique` est explicitement exclue : elle est calculée à partir du prix
# lui-même (cf. preprocessing.flag_prix_extremes), donc l'utiliser comme feature
# d'entraînement serait une fuite de données (data leakage) — et de toute façon
# impossible à fournir honnêtement à l'API pour une nouvelle annonce, puisqu'on
# ne connaît pas encore son prix (c'est justement ce qu'on prédit).
COLONNES_NON_FEATURES = ["id", "titre", "adresse", "date_publication", "prix_atypique"]

TARGET_COLUMN = "prix_loyer_mensuel"


def parse_equipements(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforme la colonne `equipements` (chaîne "piscine|parking|...") en
    autant de colonnes binaires (`equip_piscine`, `equip_parking`, ...).
    """
    df = df.copy()
    equipements_lists = (
        df["equipements"].fillna("").apply(lambda s: set(s.split("|")) if s else set())
    )

    for equip in EQUIPEMENTS_CONNUS:
        df[f"equip_{equip}"] = equipements_lists.apply(lambda s: equip in s)

    df = df.drop(columns=["equipements"])
    return df


def encode_categorical(
    df: pd.DataFrame, columns: Sequence[str] = ("ville", "type_bien")
) -> pd.DataFrame:
    """
    One-hot encode les variables catégorielles à faible cardinalité.

    `quartier` est volontairement exclu de cet encodage simple (trop de
    modalités différentes pour un one-hot direct) ; à traiter séparément
    si besoin (frequency encoding, regroupement manuel...), documenté comme
    piste d'amélioration dans le rapport.
    """
    df = df.copy()
    existing_columns = [c for c in columns if c in df.columns]
    df = pd.get_dummies(df, columns=existing_columns, drop_first=True)
    return df


def select_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Retire les colonnes non pertinentes pour l'entraînement (id, texte libre...)."""
    df = df.copy()
    columns_to_drop = [c for c in COLONNES_NON_FEATURES if c in df.columns]
    if "quartier" in df.columns:
        columns_to_drop.append("quartier")
    return df.drop(columns=columns_to_drop)


def align_features(df: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    """
    Aligne les colonnes de `df` sur exactement celles vues à l'entraînement
    (`feature_columns`), dans le même ordre. Les colonnes manquantes (ex. une
    ville qui n'a pas été one-hot encodée car absente du batch d'entraînement)
    sont ajoutées avec la valeur 0/False. Indispensable pour l'inférence via
    l'API : le modèle attend un vecteur de features de forme identique à celle
    utilisée pendant `fit()`.
    """
    return df.reindex(columns=list(feature_columns), fill_value=False)


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Pipeline complet de feature engineering : df nettoyé -> (X, y) prêts pour
    l'entraînement. `y` est la variable cible (`prix_loyer_mensuel`), retirée de X.
    """
    df = parse_equipements(df)
    df = encode_categorical(df)
    df = select_feature_columns(df)

    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Colonne cible '{TARGET_COLUMN}' absente du DataFrame fourni.")

    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])

    logger.info("Feature matrix construite : %d lignes, %d colonnes", len(X), X.shape[1])
    return X, y
