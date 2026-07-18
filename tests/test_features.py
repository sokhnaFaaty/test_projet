"""Tests unitaires pour senegal_rental_price.features.build_features."""

import pandas as pd
import pytest

from senegal_rental_price.features.build_features import (
    build_feature_matrix,
    encode_categorical,
    parse_equipements,
    select_feature_columns,
)


@pytest.fixture
def cleaned_df() -> pd.DataFrame:
    """Simule une sortie typique de preprocessing.clean_pipeline()."""
    return pd.DataFrame(
        {
            "id": ["a1", "a2", "a3"],
            "ville": ["Dakar", "Thiès", "Dakar"],
            "quartier": ["Almadies", "Non renseigné", "Mermoz"],
            "type_bien": ["appartement", "maison", "maison"],
            "surface_m2": [80.0, 200.0, 120.0],
            "surface_estimee": [False, True, True],
            "nb_pieces": [3.0, 5.0, 4.0],
            "nb_pieces_imputee": [False, False, True],
            "nb_chambres": [2.0, 4.0, 3.0],
            "nb_chambres_imputee": [False, False, False],
            "meuble": [True, False, True],
            "equipements": ["piscine|parking", "gardiennage", ""],
            "prix_atypique": [False, False, False],
            "prix_loyer_mensuel": [900_000, 1_200_000, 700_000],
            "titre": ["Bel appart", "Villa", "Studio"],
            "adresse": ["Almadies, Dakar", None, "Mermoz"],
            "date_publication": ["2026-01-01", "2026-01-02", "2026-01-03"],
        }
    )


class TestParseEquipements:
    def test_creates_binary_columns_for_known_equipements(self, cleaned_df: pd.DataFrame) -> None:
        result = parse_equipements(cleaned_df)
        assert "equip_piscine" in result.columns
        assert "equip_parking" in result.columns
        assert "equip_gardiennage" in result.columns

    def test_binary_columns_have_correct_values(self, cleaned_df: pd.DataFrame) -> None:
        result = parse_equipements(cleaned_df)
        assert result.loc[0, "equip_piscine"] == True  # noqa: E712
        assert result.loc[0, "equip_gardiennage"] == False  # noqa: E712
        assert result.loc[1, "equip_gardiennage"] == True  # noqa: E712

    def test_empty_equipements_gives_all_false(self, cleaned_df: pd.DataFrame) -> None:
        result = parse_equipements(cleaned_df)
        assert result.loc[2, "equip_piscine"] == False  # noqa: E712
        assert result.loc[2, "equip_gardiennage"] == False  # noqa: E712

    def test_original_equipements_column_is_dropped(self, cleaned_df: pd.DataFrame) -> None:
        result = parse_equipements(cleaned_df)
        assert "equipements" not in result.columns


class TestEncodeCategorical:
    def test_creates_dummy_columns_for_ville(self, cleaned_df: pd.DataFrame) -> None:
        result = encode_categorical(cleaned_df)
        assert any(col.startswith("ville_") for col in result.columns)
        assert "ville" not in result.columns

    def test_creates_dummy_columns_for_type_bien(self, cleaned_df: pd.DataFrame) -> None:
        result = encode_categorical(cleaned_df)
        assert any(col.startswith("type_bien_") for col in result.columns)
        assert "type_bien" not in result.columns

    def test_quartier_is_not_encoded_by_default(self, cleaned_df: pd.DataFrame) -> None:
        result = encode_categorical(cleaned_df)
        assert "quartier" in result.columns

    def test_drop_first_avoids_perfect_collinearity(self, cleaned_df: pd.DataFrame) -> None:
        result = encode_categorical(cleaned_df)
        # Avec drop_first=True, une catégorie de ville est utilisée comme référence
        # (donc n_valeurs_uniques - 1 colonnes créées, pas n_valeurs_uniques).
        n_villes_uniques = cleaned_df["ville"].nunique()
        n_colonnes_ville = sum(1 for col in result.columns if col.startswith("ville_"))
        assert n_colonnes_ville == n_villes_uniques - 1


class TestSelectFeatureColumns:
    def test_drops_non_feature_columns(self, cleaned_df: pd.DataFrame) -> None:
        result = select_feature_columns(cleaned_df)
        for col in ["id", "titre", "adresse", "date_publication", "quartier"]:
            assert col not in result.columns

    def test_drops_prix_atypique_to_avoid_leakage(self, cleaned_df: pd.DataFrame) -> None:
        result = select_feature_columns(cleaned_df)
        assert "prix_atypique" not in result.columns

    def test_keeps_relevant_columns(self, cleaned_df: pd.DataFrame) -> None:
        result = select_feature_columns(cleaned_df)
        for col in ["surface_m2", "nb_pieces", "prix_loyer_mensuel", "meuble"]:
            assert col in result.columns


class TestBuildFeatureMatrix:
    def test_returns_x_and_y_with_matching_length(self, cleaned_df: pd.DataFrame) -> None:
        X, y = build_feature_matrix(cleaned_df)
        assert len(X) == len(y) == len(cleaned_df)

    def test_target_column_not_in_features(self, cleaned_df: pd.DataFrame) -> None:
        X, _ = build_feature_matrix(cleaned_df)
        assert "prix_loyer_mensuel" not in X.columns

    def test_target_values_are_correct(self, cleaned_df: pd.DataFrame) -> None:
        _, y = build_feature_matrix(cleaned_df)
        assert list(y) == [900_000, 1_200_000, 700_000]

    def test_raises_if_target_column_missing(self, cleaned_df: pd.DataFrame) -> None:
        df_sans_cible = cleaned_df.drop(columns=["prix_loyer_mensuel"])
        with pytest.raises(KeyError):
            build_feature_matrix(df_sans_cible)

    def test_no_identifier_or_text_columns_in_final_features(
        self, cleaned_df: pd.DataFrame
    ) -> None:
        X, _ = build_feature_matrix(cleaned_df)
        for col in ["id", "titre", "adresse", "date_publication", "quartier"]:
            assert col not in X.columns
