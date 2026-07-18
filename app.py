"""
Scraper NeoBien - récupère les annonces de location (appartement + maison)
via l'API JSON interne du site, découverte via DevTools (onglet Network > Fetch/XHR).

Endpoint : https://neobien.com/api/realEstate?type=location&propertyType={appartement|maison}&page={n}&limit=12

Usage strictement académique - projet M2 DSIA.
"""

import csv
import random
import sys
import time
from typing import Any

import requests

# Windows (PowerShell/cmd) utilise souvent cp1252 par défaut, incapable d'afficher
# certains emojis présents dans les annonces (🏡, 🌊...). On force l'UTF-8.
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

BASE_URL = "https://neobien.com/api/realEstate"
HEADERS = {
    # Un vrai User-Agent de navigateur : l'API bloque/renvoie une réponse
    # différente pour un UA générique de script.
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
    # Referer/Origin imitent une requête faite depuis la page d'annonces elle-même.
    "Referer": "https://neobien.com/annonces?type=location&category=maison",
    "Origin": "https://neobien.com",
}
LIMIT = 12
SLEEP_BETWEEN_REQUESTS = 1.0  # politesse envers le serveur, à ne pas descendre trop bas
OUTPUT_CSV = "locations.csv"
DEBUG = False  # passe à True pour déboguer les réponses brutes de l'API

# Mots-clés simples pour détecter des équipements dans le texte libre (title + description).
# Heuristique volontairement basique : à documenter comme limitation dans le rapport.
EQUIPMENT_KEYWORDS = {
    "piscine": ["piscine"],
    "climatisation": ["climatis", "clim "],
    "gardiennage": ["gardien", "sécurisé", "securise"],
    "parking": ["parking", "garage"],
    "meuble": ["meublé", "meuble"],
    "wifi": ["wifi", "fibre"],
    "jardin": ["jardin"],
    "terrasse": ["terrasse"],
    "salle_de_sport": ["salle de sport", "gym"],
}


def fetch_page(property_type: str, page: int) -> dict[str, Any]:
    """Récupère une page de résultats pour un type de bien donné."""
    params = {
        "type": "location",
        "propertyType": property_type,
        "page": page,
        "limit": LIMIT,
    }
    response = requests.get(BASE_URL, headers=HEADERS, params=params, timeout=15)

    if DEBUG:
        print(f"--- DEBUG {property_type} page {page} ---")
        print("URL appelée :", response.url)
        print("Status code :", response.status_code)
        print("Content-Type:", response.headers.get("Content-Type"))
        print("Premiers 500 caractères de la réponse brute :")
        print(response.text[:500])
        print("--- FIN DEBUG ---")

    response.raise_for_status()
    return response.json()


def extract_equipements(item: dict[str, Any]) -> list[str]:
    """Détection heuristique d'équipements à partir du titre + de la description."""
    text = f"{item.get('title', '')} {item.get('description', '')}".lower()
    found = []
    for label, patterns in EQUIPMENT_KEYWORDS.items():
        if any(p in text for p in patterns):
            found.append(label)
    return found


def generate_synthetic_surface(item: dict[str, Any], property_type: str) -> tuple[float, bool]:
    """
    Retourne (surface_m2, surface_estimee).

    Si l'API fournit une surface valide (> 0), on la garde telle quelle
    (surface_estimee=False). Sinon, on génère une estimation réaliste basée
    sur le nombre de pièces et le type de bien, avec une variabilité aléatoire
    mais DÉTERMINISTE (seedée sur l'id de l'annonce) : relancer le scraper sur
    la même annonce donnera toujours la même surface estimée.
    """
    raw_surface = item.get("property_surface")
    if raw_surface and isinstance(raw_surface, (int, float)) and raw_surface > 5:
        return float(raw_surface), False

    rooms = item.get("rooms") or item.get("bedrooms") or 2
    rooms = max(1, int(rooms))

    if property_type == "appartement":
        base, per_room = 15, 18
    else:  # maison, villa, studio classé "maison" dans l'API
        base, per_room = 25, 30

    estimated = base + per_room * rooms

    rng = random.Random(item.get("_id", str(rooms)))
    noise_factor = rng.uniform(0.85, 1.25)
    estimated *= noise_factor

    estimated = round(max(15.0, min(estimated, 1500.0)), 1)
    return estimated, True


def parse_annonce(item: dict[str, Any], property_type: str) -> dict[str, Any] | None:
    """
    Transforme une entrée brute de l'API vers le schéma attendu par le projet.
    Retourne None si l'annonce doit être exclue (prix invalide, location saisonnière...).
    """
    if item.get("rentalPeriod") != "monthly":
        return None

    price = item.get("price")
    if not price or price <= 0:
        return None

    surface_m2, surface_estimee = generate_synthetic_surface(item, property_type)

    return {
        "id": item.get("_id"),
        "ville": item.get("region"),
        "quartier": item.get("city"),
        "type_bien": property_type,
        "surface_m2": surface_m2,
        "surface_estimee": surface_estimee,
        "nb_pieces": item.get("rooms"),
        "nb_chambres": item.get("bedrooms"),
        "meuble": item.get("furnished"),
        "equipements": extract_equipements(item),
        "prix_loyer_mensuel": price,
        "titre": item.get("title"),
        "adresse": item.get("address"),
        "date_publication": item.get("createdAt"),
    }


def scrape_property_type(property_type: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    page = 1
    total_pages = None

    while total_pages is None or page <= total_pages:
        data = fetch_page(property_type, page)
        items = data.get("data", [])
        meta = data.get("meta", {})
        total_pages = meta.get("totalPages", page)

        if not items:
            print(f"Aucune annonce trouvée à la page {page} pour {property_type}")
            break

        kept = 0
        for item in items:
            parsed = parse_annonce(item, property_type)
            if parsed is not None:
                results.append(parsed)
                kept += 1

        print(
            f"{property_type} - page {page}/{total_pages}: "
            f"{len(items)} annonces reçues, {kept} conservées"
        )
        page += 1
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    return results


def main() -> None:
    all_results: list[dict[str, Any]] = []
    for property_type in ["appartement", "maison"]:
        all_results.extend(scrape_property_type(property_type))

    if not all_results:
        print("Scraping terminé ! 0 annonces enregistrées dans locations.csv")
        return

    fieldnames = list(all_results[0].keys())
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        for row in all_results:
            row = dict(row)
            row["equipements"] = "|".join(row["equipements"])
            writer.writerow(row)

    print(f"Scraping terminé ! {len(all_results)} annonces enregistrées dans {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
