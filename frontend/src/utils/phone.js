/**
 * Utilitaires téléphone (format E.164 simplifié).
 * Backend impose: 7 à 15 chiffres, option '+' en tête.
 */

/** Regex alignée backend: ^\+?\d{7,15}$ */
export const PHONE_REGEX = /^\+?\d{7,15}$/;

/** Message d'erreur user-friendly pour la validation UI */
export const PHONE_VALIDATION_MESSAGE =
  "Format attendu : +41791234567 ou 0791234567 (7 à 15 chiffres, + optionnel).";

/**
 * Normalise une chaîne téléphone pour envoi API.
 * - trim
 * - supprime espaces, tirets, parenthèses, points
 * - vide => null (ne pas envoyer "")
 * - accepte '+' optionnel en tête
 * - 00 au début => converti en +
 * - ne garde que chiffres + '+' en première position
 *
 * @param {string | null | undefined} raw - Valeur brute du champ
 * @returns {string | null} - Valeur normalisée ou null si vide
 */
export function normalizePhone(raw) {
  if (raw == null) return null;
  let s = String(raw).trim();
  if (s === "") return null;
  // Remplacer 00 au début par +
  if (s.startsWith("00")) {
    s = "+" + s.slice(2);
  }
  // Garder uniquement chiffres et + en première position
  let hasPlus = s.startsWith("+");
  let digits = s.replace(/\D/g, "");
  if (digits.length === 0) return null;
  // Format suisse/européen: +41 (0)79... => supprimer le 0 après l'indicatif pays (2 chiffres)
  if (hasPlus && digits.length > 3 && digits[2] === "0") {
    digits = digits.slice(0, 2) + digits.slice(3);
  }
  return hasPlus ? "+" + digits : digits;
}

/**
 * Valide une chaîne normalisée (format backend).
 *
 * @param {string | null} normalized - Valeur déjà normalisée
 * @returns {boolean} - true si valide ou vide
 */
export function isValidPhone(normalized) {
  if (normalized == null || normalized === "") return true;
  return PHONE_REGEX.test(normalized);
}

/**
 * Valide et retourne un message d'erreur ou null.
 *
 * @param {string | null} normalized - Valeur normalisée
 * @returns {string | null} - Message d'erreur ou null si valide
 */
export function getPhoneValidationError(normalized) {
  if (normalized == null || normalized === "") return null;
  if (PHONE_REGEX.test(normalized)) return null;
  return PHONE_VALIDATION_MESSAGE;
}
