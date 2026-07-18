import { useState } from "react";
import { predictPrice, EQUIPEMENTS_DISPONIBLES, VILLES } from "./api.js";

const TYPES_BIEN = [
  { value: "appartement", label: "Appartement" },
  { value: "maison", label: "Maison" },
];

const INITIAL_FORM = {
  ville: "Dakar",
  type_bien: "appartement",
  surface_m2: 80,
  nb_pieces: 3,
  nb_chambres: 2,
  meuble: false,
  equipements: [],
};

function formatValidationErrors(detail) {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((err) => {
        const field = err.loc?.[err.loc.length - 1] ?? "champ";
        return `${field} : ${err.msg}`;
      })
      .join(" — ");
  }
  return "Une erreur est survenue.";
}

function formatFCFA(value) {
  return new Intl.NumberFormat("fr-FR").format(value);
}

export default function App() {
  const [form, setForm] = useState(INITIAL_FORM);
  const [status, setStatus] = useState("idle"); // idle | loading | success | error
  const [result, setResult] = useState(null);
  const [errorMessage, setErrorMessage] = useState("");

  function updateField(field, value) {
    setForm((prev) => ({ ...prev, [field]: value }));
  }

  function toggleEquipement(value) {
    setForm((prev) => {
      const already = prev.equipements.includes(value);
      return {
        ...prev,
        equipements: already
          ? prev.equipements.filter((e) => e !== value)
          : [...prev.equipements, value],
      };
    });
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setStatus("loading");
    setErrorMessage("");

    try {
      const payload = {
        ...form,
        surface_m2: Number(form.surface_m2),
        nb_pieces: Number(form.nb_pieces),
        nb_chambres: Number(form.nb_chambres),
      };
      const prediction = await predictPrice(payload);
      setResult(prediction);
      setStatus("success");
    } catch (err) {
      setErrorMessage(formatValidationErrors(err.detail));
      setStatus("error");
    }
  }

  return (
    <div className="page">
      <header className="page-header">
        <span className="eyebrow">Sénégal · estimation immobilière</span>
        <h1>Combien vaut ce bien à louer ?</h1>
        <p className="lede">
          Renseignez les caractéristiques du logement pour obtenir une estimation
          du loyer mensuel, calculée par un modèle entraîné sur des annonces réelles.
        </p>
      </header>

      <div className="layout">
        <form className="panel form-panel" onSubmit={handleSubmit}>
          <div className="field-row">
            <label className="field">
              <span>Ville</span>
              <select value={form.ville} onChange={(e) => updateField("ville", e.target.value)}>
                {VILLES.map((v) => (
                  <option key={v} value={v}>
                    {v}
                  </option>
                ))}
              </select>
            </label>

            <label className="field">
              <span>Type de bien</span>
              <select
                value={form.type_bien}
                onChange={(e) => updateField("type_bien", e.target.value)}
              >
                {TYPES_BIEN.map((t) => (
                  <option key={t.value} value={t.value}>
                    {t.label}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <label className="field">
            <span>Surface habitable (m²)</span>
            <input
              type="number"
              min="1"
              max="2000"
              step="1"
              value={form.surface_m2}
              onChange={(e) => updateField("surface_m2", e.target.value)}
              required
            />
          </label>

          <div className="field-row">
            <label className="field">
              <span>Nombre de pièces</span>
              <input
                type="number"
                min="1"
                max="20"
                value={form.nb_pieces}
                onChange={(e) => updateField("nb_pieces", e.target.value)}
                required
              />
            </label>

            <label className="field">
              <span>Nombre de chambres</span>
              <input
                type="number"
                min="0"
                max="15"
                value={form.nb_chambres}
                onChange={(e) => updateField("nb_chambres", e.target.value)}
                required
              />
            </label>
          </div>

          <label className="field field-checkbox">
            <input
              type="checkbox"
              checked={form.meuble}
              onChange={(e) => updateField("meuble", e.target.checked)}
            />
            <span>Le bien est meublé</span>
          </label>

          <fieldset className="field">
            <legend>Équipements</legend>
            <div className="equip-grid">
              {EQUIPEMENTS_DISPONIBLES.map((eq) => (
                <label key={eq.value} className="equip-chip">
                  <input
                    type="checkbox"
                    checked={form.equipements.includes(eq.value)}
                    onChange={() => toggleEquipement(eq.value)}
                  />
                  <span>{eq.label}</span>
                </label>
              ))}
            </div>
          </fieldset>

          <button type="submit" className="submit-btn" disabled={status === "loading"}>
            {status === "loading" ? "Estimation en cours…" : "Estimer le loyer"}
          </button>

          {status === "error" && (
            <p className="error-message" role="alert">
              {errorMessage}
            </p>
          )}
        </form>

        <aside className="panel ticket-panel" aria-live="polite">
          {status !== "success" ? (
            <div className="ticket-empty">
              <p>Remplissez le formulaire pour voir apparaître ici l'estimation du loyer mensuel.</p>
            </div>
          ) : (
            <div className="ticket">
              <span className="ticket-eyebrow">Estimation du loyer mensuel</span>
              <p className="ticket-amount">
                {formatFCFA(result.prix_loyer_mensuel_estime)}
                <span className="ticket-currency">{result.devise}</span>
              </p>
              <div className="ticket-divider" aria-hidden="true" />
              <dl className="ticket-details">
                <div>
                  <dt>Ville</dt>
                  <dd>{form.ville}</dd>
                </div>
                <div>
                  <dt>Bien</dt>
                  <dd>
                    {form.type_bien} · {form.surface_m2} m²
                  </dd>
                </div>
                <div>
                  <dt>Modèle</dt>
                  <dd>v{result.model_version}</dd>
                </div>
              </dl>
              <p className="ticket-disclaimer">
                Estimation indicative, basée sur des annonces historiques. Ne
                constitue pas une garantie de prix réel.
              </p>
            </div>
          )}
        </aside>
      </div>
    </div>
  );
}
