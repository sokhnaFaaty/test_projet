"""Tests unitaires pour senegal_rental_price.data.preprocessing."""

from pathlib import Path

import pandas as pd
import pytest

from senegal_rental_price.data.preprocessing import (
    PRIX_ALERTE_BAS,
    PRIX_ALERTE_HAUT,
    clean_pipeline,
    clean_villes,
    drop_useless_columns,
    flag_prix_extremes,
    handle_missing_values,
    load_raw_data,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Petit jeu de données représentatif, avec valeurs manquantes et anomalies volontaires."""
    return pd.DataFrame(
        {
            "id": ["a1", "a2", "a3", "a4"],
            "ville": ["Dakar", " Thiès ", "Diakhirate", "Dakar"],
            "quartier": ["Almadies", None, "Nouvel Horizon", "Mermoz"],
            "type_bien": ["appartement", "maison", "appartement", "maison"],
            "surface_m2": [80.0, 120.0, 45.0, 200.0],
            "surface_estimee": [False, True, True, False],
            "nb_pieces": [3.0, None, 2.0, 5.0],
            "nb_chambres": [2.0, 3.0, None, 4.0],
            "meuble": [True, None, False, True],
            "equipements": ["piscine|parking", None, "gardiennage", ""],
            "prix_loyer_mensuel": [900_000, 3_500, 700_000, 15_000_000],
            "titre": ["Bel appart", None, "Studio", "Villa"],
            "adresse": [None, "Thiès centre", None, "Mermoz"],
            "date_publication": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
        }
    )


class TestLoadRawData:
    def test_load_raw_data_reads_csv_correctly(
        self, tmp_path: Path, sample_df: pd.DataFrame
    ) -> None:
        csv_path = tmp_path / "raw.csv"
        sample_df.to_csv(csv_path, sep=";", index=False, encoding="utf-8-sig")

        result = load_raw_data(csv_path)

        assert len(result) == len(sample_df)
        assert list(result.columns) == list(sample_df.columns)


class TestCleanVilles:
    def test_clean_villes_does_not_drop_any_row(self, sample_df: pd.DataFrame) -> None:
        result = clean_villes(sample_df)
        assert len(result) == len(sample_df)

    def test_clean_villes_strips_whitespace(self, sample_df: pd.DataFrame) -> None:
        result = clean_villes(sample_df)
        assert result.loc[1, "ville"] == "Thiès"

    def test_clean_villes_keeps_rare_values(self, sample_df: pd.DataFrame) -> None:
        result = clean_villes(sample_df)
        assert "Diakhirate" in result["ville"].values


class TestFlagPrixExtremes:
    def test_flag_prix_extremes_does_not_drop_any_row(self, sample_df: pd.DataFrame) -> None:
        result = flag_prix_extremes(sample_df)
        assert len(result) == len(sample_df)

    def test_flag_prix_extremes_flags_out_of_range_values(self, sample_df: pd.DataFrame) -> None:
        result = flag_prix_extremes(sample_df)
        # 3 500 FCFA et 15 000 000 FCFA sont hors de [PRIX_ALERTE_BAS, PRIX_ALERTE_HAUT]
        assert result.loc[1, "prix_atypique"] == True  # noqa: E712
        assert result.loc[3, "prix_atypique"] == True  # noqa: E712

    def test_flag_prix_extremes_does_not_flag_normal_values(self, sample_df: pd.DataFrame) -> None:
        result = flag_prix_extremes(sample_df)
        assert result.loc[0, "prix_atypique"] == False  # noqa: E712
        assert result.loc[2, "prix_atypique"] == False  # noqa: E712

    def test_prix_alerte_bounds_are_coherent(self) -> None:
        assert PRIX_ALERTE_BAS < PRIX_ALERTE_HAUT


class TestDropUselessColumns:
    def test_drops_adresse_column(self, sample_df: pd.DataFrame) -> None:
        result = drop_useless_columns(sample_df)
        assert "adresse" not in result.columns

    def test_does_not_drop_any_row(self, sample_df: pd.DataFrame) -> None:
        result = drop_useless_columns(sample_df)
        assert len(result) == len(sample_df)

    def test_keeps_other_columns_intact(self, sample_df: pd.DataFrame) -> None:
        result = drop_useless_columns(sample_df)
        assert "ville" in result.columns
        assert "quartier" in result.columns


class TestHandleMissingValues:
    def test_no_missing_values_after_cleaning(self, sample_df: pd.DataFrame) -> None:
        result = handle_missing_values(sample_df)
        assert result["nb_pieces"].isna().sum() == 0
        assert result["nb_chambres"].isna().sum() == 0
        assert result["meuble"].isna().sum() == 0
        assert result["equipements"].isna().sum() == 0
        assert result["quartier"].isna().sum() == 0
        assert result["titre"].isna().sum() == 0

    def test_imputation_flags_are_correct(self, sample_df: pd.DataFrame) -> None:
        result = handle_missing_values(sample_df)
        assert result.loc[1, "nb_pieces_imputee"] == True  # noqa: E712
        assert result.loc[0, "nb_pieces_imputee"] == False  # noqa: E712
        assert result.loc[2, "nb_chambres_imputee"] == True  # noqa: E712

    def test_meuble_defaults_to_false(self, sample_df: pd.DataFrame) -> None:
        result = handle_missing_values(sample_df)
        assert result.loc[1, "meuble"] == False  # noqa: E712

    def test_missing_text_fields_replaced_with_placeholder(self, sample_df: pd.DataFrame) -> None:
        result = handle_missing_values(sample_df)
        assert result.loc[1, "quartier"] == "Non renseigné"
        assert result.loc[1, "titre"] == "Non renseigné"

    def test_missing_equipements_becomes_empty_string(self, sample_df: pd.DataFrame) -> None:
        result = handle_missing_values(sample_df)
        assert result.loc[1, "equipements"] == ""

    def test_nb_pieces_imputed_with_median_by_type_bien(self, sample_df: pd.DataFrame) -> None:
        # maison : valeurs connues [3.0 (nb_chambres index3=4), nb_pieces connu index1=None ->
        # médiane des nb_pieces "maison" (index1=None, index3=5.0) -> seule valeur connue = 5.0
        result = handle_missing_values(sample_df)
        assert result.loc[1, "nb_pieces"] == 5.0


class TestCleanPipeline:
    def test_clean_pipeline_creates_output_file(
        self, tmp_path: Path, sample_df: pd.DataFrame
    ) -> None:
        raw_path = tmp_path / "raw.csv"
        processed_path = tmp_path / "processed" / "locations_clean.csv"
        sample_df.to_csv(raw_path, sep=";", index=False, encoding="utf-8-sig")

        result = clean_pipeline(raw_path, processed_path)

        assert processed_path.exists()
        assert len(result) == len(sample_df)

    def test_clean_pipeline_preserves_all_rows(
        self, tmp_path: Path, sample_df: pd.DataFrame
    ) -> None:
        raw_path = tmp_path / "raw.csv"
        processed_path = tmp_path / "processed" / "locations_clean.csv"
        sample_df.to_csv(raw_path, sep=";", index=False, encoding="utf-8-sig")

        result = clean_pipeline(raw_path, processed_path)

        # Décision produit : aucune ligne n'est jamais exclue par ce pipeline.
        assert len(result) == len(sample_df)

    def test_clean_pipeline_drops_adresse_column(
        self, tmp_path: Path, sample_df: pd.DataFrame
    ) -> None:
        raw_path = tmp_path / "raw.csv"
        processed_path = tmp_path / "processed" / "locations_clean.csv"
        sample_df.to_csv(raw_path, sep=";", index=False, encoding="utf-8-sig")

        result = clean_pipeline(raw_path, processed_path)

        assert "adresse" not in result.columns

    def test_clean_pipeline_output_has_no_missing_values_in_key_columns(
        self, tmp_path: Path, sample_df: pd.DataFrame
    ) -> None:
        raw_path = tmp_path / "raw.csv"
        processed_path = tmp_path / "processed" / "locations_clean.csv"
        sample_df.to_csv(raw_path, sep=";", index=False, encoding="utf-8-sig")

        result = clean_pipeline(raw_path, processed_path)

        for col in ["nb_pieces", "nb_chambres", "meuble", "equipements", "quartier"]:
            assert result[col].isna().sum() == 0
