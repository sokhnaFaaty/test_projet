# Données — prédiction du prix des locations au Sénégal 

## Provenance

Les données ont été collectées par **scraping** de la plateforme immobilière
[NeoBien](https://neobien.com), via son API JSON interne
(`https://neobien.com/api/realEstate`), découverte via l'onglet Network des
DevTools du navigateur. Usage strictement académique, dans le cadre du projet
M2 DSIA.

- Catégories collectées : annonces de location, types `appartement` et `maison`.
- Date de collecte : 07/07/2026.
- Script de collecte : `scripts/scrape_neobien.py` (hors `src/`, non soumis aux
  exigences de typage/tests strictes de la section 3 du sujet).
- Volume brut collecté : 166 annonces (appartement, maison).

## Description des variables (données brutes, `data/raw/locations.csv`)

| Variable            | Description                                                                 |
|---------------------|-------------------------------------------------------------------------------|
| `id`                | Identifiant unique de l'annonce (API NeoBien)                                |
| `ville`             | Ville/région au sens large (Dakar, Thiès, Ziguinchor...)                     |
| `quartier`          | Quartier ou commune précise (Almadies, Mermoz, Ngaparou...)                  |
| `type_bien`         | `appartement` ou `maison`                                                    |
| `surface_m2`        | Surface en m² — réelle si fournie par l'API, sinon **estimée** (voir ci-dessous) |
| `surface_estimee`   | Booléen : `True` si `surface_m2` a été générée, `False` si issue de l'API    |
| `nb_pieces`         | Nombre de pièces                                                             |
| `nb_chambres`       | Nombre de chambres                                                           |
| `meuble`            | Booléen — meublé ou non                                                      |
| `equipements`       | Liste d'équipements détectés dans le titre/description, séparés par `\|`     |
| `prix_loyer_mensuel`| Variable cible, en FCFA (uniquement les locations mensuelles classiques)     |
| `titre`, `adresse`, `date_publication` | Champs bruts conservés pour traçabilité, non utilisés pour le modèle |

## Limites connues

- **Surfaces majoritairement estimées** : l'API NeoBien ne renseigne pas
  systématiquement `property_surface`. Quand absente, une estimation est
  générée à partir du nombre de pièces et du type de bien (voir
  `generate_synthetic_surface()` dans le script de scraping), avec un bruit
  aléatoire déterministe. Proportion réelle vs estimée : voir
  `notebooks/01_exploration.ipynb`.
- **Équipements extraits heuristiquement** : détection par mots-clés dans le
  titre/description (pas de champ structuré côté source), donc possibles
  faux négatifs/positifs.
- **Locations saisonnières exclues** : les annonces avec `rentalPeriod` autre
  que `monthly` (locations journalières type Airbnb) ont été retirées pour ne
  pas fausser `prix_loyer_mensuel`.
- **Annonces à prix nul exclues** (probablement incomplètes côté source).
- **Qualité variable de la source** : certaines annonces présentent des
  incohérences (ex. ville renseignée avec une valeur hors de la liste
  attendue), corrigées/filtrées lors du nettoyage (`preprocessing.py`),
  documenté dans le notebook d'exploration.

## Fichiers

- `raw/locations.csv` — export brut du scraping (non versionné, voir `.gitignore`)
- `processed/locations_clean.csv` — après passage par `preprocessing.clean_pipeline()`