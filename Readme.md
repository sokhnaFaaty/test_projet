# Prédiction du prix des locations au Sénégal 



Projet M2 DSIA — mise en production d'un modèle de prédiction du prix de

location de biens immobiliers (appartement, maison) au Sénégal, de la collecte

des données jusqu'à l'exposition via une API conteneurisée.



## Installation



```bash

python -m venv venv

# Windows

venv\Scripts\activate

# Linux/Mac

source venv/bin/activate



pip install -e ".[dev]"

```



## Récupérer les données



Les données brutes ne sont pas versionnées (voir `.gitignore`). Pour les régénérer :



```bash

python scripts/scrape_neobien.py

```



Ceci génère `data/raw/locations.csv` à partir de l'API publique de

[NeoBien](https://neobien.com) (usage strictement académique).



Puis, pour nettoyer les données brutes :



```bash

python -m senegal_rental_price.data.preprocessing

```



Ceci génère `data/processed/locations_clean.csv`. Voir `data/README.md` pour

le détail de la provenance et des choix de nettoyage, et

`notebooks/01_exploration.ipynb` pour l'analyse exploratoire complète.



## Entraîner un modèle



*(à compléter une fois `train.py` et la config Hydra en place)*



```bash

python -m senegal_rental_price.models.train model=random_forest

```



## Lancer l'API



*(à compléter une fois l'API FastAPI en place)*



```bash

uvicorn api.main:app --reload

```



Documentation interactive ensuite disponible sur `http://localhost:8000/docs`.



## Lancer le front



*(à compléter une fois le front en place)*



## Lancer les tests



```bash

pytest --cov=src --cov-report=term-missing

```



## Structure du projet



Voir l'arborescence complète dans le sujet du projet. Résumé :

- `data/` — données brutes et nettoyées (non versionnées)

- `notebooks/` — analyse exploratoire

- `src/senegal_rental_price/` — code de production (typé, testé)

- `api/` — API FastAPI

- `frontend/` — interface utilisateur

- `tests/` — tests unitaires

- `docker/` — conteneurisation



## Statut CI



*(badge à ajouter une fois le pipeline GitHub Actions en place)*"# test_projet" 
"# test_projet" 
