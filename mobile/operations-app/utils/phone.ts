/**
 * Utilitaires téléphone (format E.164 simplifié).
 * Backend impose: 7 à 15 chiffres, option '+' en tête.
 * Utilisé pour le bouton "Appeler" (priorité: contact_phone > phone > gp_phone).
 */

/** Regex alignée backend: ^\+?\d{7,15}$ */
export const PHONE_REGEX = /^\+?\d{7,15}$/;

/**
 * Normalise une chaîne téléphone.
 * - trim, supprime espaces/tirets/parenthèses/points
 * - vide => null
 * - 00 au début => +
 * - ne garde que chiffres + '+' en première position
 * - +41 (0)79... => +4179...
 */
export function normalizePhone(raw: string | null | undefined): string | null {
  if (raw == null) return null;
  let s = String(raw).trim();
  if (s === "") return null;
  if (s.startsWith("00")) s = "+" + s.slice(2);
  const hasPlus = s.startsWith("+");
  let digits = s.replace(/\D/g, "");
  if (digits.length === 0) return null;
  if (hasPlus && digits.length > 3 && digits[2] === "0") {
    digits = digits.slice(0, 2) + digits.slice(3);
  }
  return hasPlus ? "+" + digits : digits;
}

/**
 * Valide une chaîne normalisée (format backend).
 */
export function isValidPhone(normalized: string | null | undefined): boolean {
  if (normalized == null || normalized === "") return false;
  return PHONE_REGEX.test(normalized);
}

/** Objet avec champs téléphone possibles (client ou mission) */
export type PhoneSource = {
  contact_phone?: string | null;
  phone?: string | null;
  gp_phone?: string | null;
  /** @deprecated Utiliser client.contact_phone */
  client_phone?: string | null;
  client?: {
    contact_phone?: string | null;
    phone?: string | null;
    gp_phone?: string | null;
  } | null;
};

/**
 * Retourne le premier numéro appelable valide selon la priorité:
 * 1. contact_phone, 2. phone, 3. gp_phone (+ client_phone legacy au niveau mission).
 * Retourne null si aucun numéro valide.
 */
export function getCallablePhone(source: PhoneSource | null | undefined): string | null {
  if (source == null) return null;

  const client = source.client;
  // Ordre officiel d'abord (contact_phone > phone > gp_phone), puis champs racine, puis legacy client_phone
  const candidates: (string | null | undefined)[] = [
    client?.contact_phone,
    client?.phone,
    client?.gp_phone,
    source.contact_phone,
    source.phone,
    source.gp_phone,
    (source as { client_phone?: string }).client_phone, // @deprecated, dernier fallback
  ];

  for (const raw of candidates) {
    const normalized = normalizePhone(raw);
    if (normalized != null && isValidPhone(normalized)) return normalized;
  }
  return null;
}
