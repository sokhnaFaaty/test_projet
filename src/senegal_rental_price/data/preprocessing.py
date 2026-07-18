"""
Fonctions de nettoyage et de prétraitement des données de location.

Ce module contient la logique de PRODUCTION (typée, testée), à la différence
du notebook 01_exploration.ipynb qui sert uniquement à explorer et justifier
les choix ci-dessous.

Décisions issues de l'exploration (notebooks/01_exploration.ipynb) :
- `ville` : aucune donnée exclue. On normalise juste la casse/les espaces, et on
  journalise les valeurs peu fréquentes ou nouvelles (ex. "Diakhirate") à titre
  informatif, sans les supprimer : ce sont des localités réelles, potentiellement
  plus représentées à mesure que le jeu de données grossit.
- `prix_loyer_mensuel` : aucune valeur exclue. Les valeurs extrêmes (ex. 3 500 FCFA
  ou 15 000 000 FCFA) sont journalisées comme avertissement mais conservées,
  l'équipe ayant jugé ces annonces plausibles au vu du marché sénégalais.
- `equipements` manquant -> liste vide plutôt que NaN.
- `nb_pieces`/`nb_chambres` manquants -> médiane, calculée séparément par `type_bien`.
- `meuble` manquant -> False par défaut (hypothèse la plus fréquente/conservatrice).
"""

from pathlib import Path

import pandas as pd

from senegal_rental_price.utils.logger import get_logger

logger = get_logger(__name__)

# Villes "attendues" au sens du sujet (Dakar, Thiès, Saint-Louis, Mbour, Saly...),
# utilisées uniquement pour de la journalisation informative, PAS pour filtrer.
VILLES_FREQUENTES = {"Dakar", "Thiès", "Saint-Louis", "Mbour", "Saly", "Ziguinchor"}

# Seuils de prix utilisés uniquement pour signaler (log) les valeurs extrêmes,
# sans les exclure : décision explicite de conserver toutes les annonces,
# jugées plausibles au vu de la diversité du marché sénégalais.
PRIX_ALERTE_BAS = 20_000
PRIX_ALERTE_HAUT = 5_000_000


def load_raw_data(path: Path) -> pd.DataFrame:
    """Charge le CSV brut issu du scraping (séparateur ';', encodage utf-8-sig)."""
    logger.info("Chargement des données brutes depuis %s", path)
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    logger.info("Données brutes chargées : %d lignes", len(df))
    return df


def clean_villes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise la colonne `ville` (espaces, casse) SANS supprimer de ligne.
    Journalise simplement les valeurs peu fréquentes pour suivi, car il peut
    s'agir de localités réelles et légitimes (ex. Diakhirate, région de Thiès).
    """
    df = df.copy()
    df["ville"] = df["ville"].str.strip()

    valeurs_rares = set(df["ville"].unique()) - VILLES_FREQUENTES
    if valeurs_rares:
        logger.info("Valeurs de ville hors de la liste habituelle (conservées) : %s", valeurs_rares)
    return df


def flag_prix_extremes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Journalise (sans les exclure) les prix hors de la plage habituelle.
    Ajoute une colonne `prix_atypique` (bool) pour permettre un traitement
    différencié ultérieur (ex. pondération, analyse séparée) sans perte de données.
    """
    df = df.copy()
    df["prix_atypique"] = ~df["prix_loyer_mensuel"].between(PRIX_ALERTE_BAS, PRIX_ALERTE_HAUT)

    nb_atypiques = int(df["prix_atypique"].sum())
    if nb_atypiques:
        logger.info(
            "%d annonce(s) avec un prix hors de [%d, %d] FCFA, conservée(s) et "
            "marquée(s) via la colonne `prix_atypique`",
            nb_atypiques,
            PRIX_ALERTE_BAS,
            PRIX_ALERTE_HAUT,
        )
    return df


def drop_useless_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supprime les colonnes trop peu renseignées pour être exploitables.

    `adresse` a été retirée : ~98% de valeurs manquantes (163/166 sur
    l'échantillon exploré dans le notebook), aucune valeur ajoutée par rapport
    à `ville`/`quartier` qui couvrent déjà la localisation.
    """
    df = df.copy()
    colonnes_a_supprimer = ["adresse"]
    existantes = [c for c in colonnes_a_supprimer if c in df.columns]
    if existantes:
        logger.info("Colonnes supprimées (trop peu renseignées) : %s", existantes)
        df = df.drop(columns=existantes)
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gère les valeurs manquantes de toutes les colonnes concernées.

    - Entiers (`nb_pieces`, `nb_chambres`) : imputés par la médiane (calculée par
      `type_bien`), JAMAIS par une valeur sentinelle comme -1 (qui fausserait
      l'apprentissage d'un modèle de régression). Une colonne booléenne
      `<col>_imputee` est ajoutée pour tracer quelles lignes ont été complétées,
      sur le même principe que `surface_estimee`.
    - Booléens (`meuble`) : False par défaut (hypothèse la plus fréquente).
    - Chaînes de caractères (`quartier`, `titre`) : "Non renseigné"
      plutôt qu'une chaîne vide ou un NaN silencieux.
    - `equipements` : liste vide (aucun équipement détecté) plutôt que NaN.
    """
    df = df.copy()

    for col in ["nb_pieces", "nb_chambres"]:
        df[f"{col}_imputee"] = df[col].isna()
        df[col] = df.groupby("type_bien")[col].transform(lambda s: s.fillna(s.median()))

    df["meuble"] = df["meuble"].astype("boolean").fillna(False).astype(bool)
    df["equipements"] = df["equipements"].fillna("")

    for col in ["quartier", "titre"]:
        if col in df.columns:
            df[col] = df[col].fillna("Non renseigné")

    return df


def clean_pipeline(raw_path: Path, processed_path: Path) -> pd.DataFrame:
    """Pipeline complet : brut -> nettoyé, avec sauvegarde du résultat. Aucune ligne exclue."""
    df = load_raw_data(raw_path)
    df = drop_useless_columns(df)
    df = clean_villes(df)
    df = flag_prix_extremes(df)
    df = handle_missing_values(df)

    before, after = len(df), len(df)  # aucune suppression de ligne dans ce pipeline
    logger.info("Pipeline de nettoyage terminé : %d -> %d lignes (aucune exclusion)", before, after)

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, sep=";", index=False, encoding="utf-8-sig")
    logger.info("Données nettoyées sauvegardées dans %s (%d lignes)", processed_path, len(df))
    return df


if __name__ == "__main__":
    clean_pipeline(
        raw_path=Path("data/raw/locations.csv"),
        processed_path=Path("data/processed/locations_clean.csv"),
    )
