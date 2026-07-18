const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const EQUIPEMENTS_DISPONIBLES = [
  { value: "piscine", label: "Piscine" },
  { value: "climatisation", label: "Climatisation" },
  { value: "gardiennage", label: "Gardiennage" },
  { value: "parking", label: "Parking" },
  { value: "wifi", label: "Wifi / fibre" },
  { value: "jardin", label: "Jardin" },
  { value: "terrasse", label: "Terrasse" },
  { value: "salle_de_sport", label: "Salle de sport" },
];

const VILLES = ["Dakar", "Thiès", "Saint-Louis", "Mbour", "Saly", "Ziguinchor"];

/**
 * Appelle POST /predict avec les caractéristiques du bien.
 * Lève une erreur enrichie (avec les détails de validation FastAPI/Pydantic
 * si disponibles) en cas d'échec, pour un affichage clair côté formulaire.
 */
async function predictPrice(payload) {
  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const body = await response.json().catch(() => null);

  if (!response.ok) {
    const error = new Error("Échec de la prédiction");
    error.status = response.status;
    error.detail = body?.detail ?? "Erreur inconnue du serveur.";
    throw error;
  }

  return body;
}

async function fetchModelInfo() {
  const response = await fetch(`${API_BASE_URL}/model/info`);
  if (!response.ok) return null;
  return response.json();
}

export { predictPrice, fetchModelInfo, EQUIPEMENTS_DISPONIBLES, VILLES, API_BASE_URL };
