/** Un seul champ interactif à la fois dans Créer une réservation. */
export type CreateRideActiveField = "client" | "pickup" | "dropoff" | "schedule" | null;

export function applyCreateRideActiveField(
  current: CreateRideActiveField,
  field: Exclude<CreateRideActiveField, null>,
  open: boolean
): CreateRideActiveField {
  if (open) return field;
  return current === field ? null : current;
}

/** Après une sélection, enchaîner le champ suivant (CREATE-RIDE-02). */
export function nextCreateRideFieldAfterSelection(
  selected: Exclude<CreateRideActiveField, null>
): CreateRideActiveField {
  if (selected === "client") return "pickup";
  if (selected === "pickup") return "dropoff";
  if (selected === "dropoff") return "schedule";
  return null;
}

export type CreateRideMissingParts = {
  hasClient: boolean;
  hasPickup: boolean;
  hasDropoff: boolean;
  hasSchedule: boolean;
  hasAmount?: boolean;
};

/** Libellé unique au-dessus de Confirmer — pas de messages d’erreur permanents. */
export function createRideMissingHint(parts: CreateRideMissingParts): string | null {
  const missing: string[] = [];
  if (!parts.hasClient) missing.push("client");
  if (!parts.hasPickup) missing.push("adresse de départ");
  if (!parts.hasDropoff) missing.push("destination");
  if (!parts.hasSchedule) missing.push("date et heure");
  if (parts.hasAmount === false) missing.push("prix");
  if (missing.length === 0) return null;
  return `À compléter : ${missing.join(", ")}`;
}
