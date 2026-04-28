/** Plafond API `client_note` — aligné `shared/client_portal_notes.py` / `BookingCreateSchema`. */
export const MAX_CLIENT_NOTE_LEN = 500;
/** Marge par ligne pour la fusion « Prise en charge » / « Destination » sous le plafond total. */
export const MAX_CLIENT_NOTE_LEG = 230;

export function formatClientDomicile(profile: {
  domicile?: { address?: string | null; zip?: string | null; city?: string | null };
} | null | undefined): string | null {
  const address = profile?.domicile?.address?.trim();
  const zip = profile?.domicile?.zip?.trim();
  const city = profile?.domicile?.city?.trim();
  const pieces = [address, [zip, city].filter(Boolean).join(" ").trim()].filter(Boolean);
  if (pieces.length === 0) return null;
  return pieces.join(", ");
}

/** Même format que le backend attend dans `client_note` (voir `compose_client_portal_notes_medical`). */
export function buildClientNoteFromLegs(departureHint: string, arrivalHint: string): string {
  const d = String(departureHint || "").trim();
  const a = String(arrivalHint || "").trim();
  if (!d && !a) return "";
  const parts = [];
  if (d) parts.push(`Prise en charge: ${d}`);
  if (a) parts.push(`Destination: ${a}`);
  return parts.join("\n").slice(0, MAX_CLIENT_NOTE_LEN);
}

export function parseCoordInput(text: string): { lat: number; lon: number } | null {
  if (!/^-?\d+(\.\d+)?,\s*-?\d+(\.\d+)?$/.test(text.trim())) return null;
  const [lat, lon] = text
    .split(",")
    .map((value) => Number(value.trim()));
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) return null;
  return { lat, lon };
}

/** Évite d'afficher une paire lat,lon comme « adresse » après autocomplete. */
export function isCoordinatePairLabel(text: string): boolean {
  return parseCoordInput(text) !== null;
}

export function formatDateYmd(value: Date): string {
  const yyyy = value.getFullYear();
  const mm = String(value.getMonth() + 1).padStart(2, "0");
  const dd = String(value.getDate()).padStart(2, "0");
  return `${yyyy}-${mm}-${dd}`;
}

export function formatTimeHm(value: Date): string {
  const hh = String(value.getHours()).padStart(2, "0");
  const min = String(value.getMinutes()).padStart(2, "0");
  return `${hh}:${min}`;
}
