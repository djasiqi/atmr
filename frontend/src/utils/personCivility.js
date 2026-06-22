/** Civilité courte (convention institution / fiches patient). */
const GENDER_SHORT = {
  HOMME: 'M.',
  MALE: 'M.',
  M: 'M.',
  FEMME: 'Mme',
  FEMALE: 'Mme',
  F: 'Mme',
};

/** Civilité longue (libellé affichage). */
const GENDER_LABEL = {
  HOMME: 'Monsieur',
  MALE: 'Monsieur',
  M: 'Monsieur',
  FEMME: 'Madame',
  FEMALE: 'Madame',
  F: 'Madame',
  AUTRE: 'Autre',
  OTHER: 'Autre',
};

function normalizeGenderKey(gender) {
  if (gender == null || gender === '') return null;
  return String(gender).trim().toUpperCase();
}

/** @returns {'M.'|'Mme'|null} */
export function getGenderShortLabel(gender) {
  const key = normalizeGenderKey(gender);
  if (!key || key === 'AUTRE' || key === 'OTHER') return null;
  return GENDER_SHORT[key] || null;
}

/** @returns {'Monsieur'|'Madame'|'Autre'|null} */
export function getGenderLabel(gender) {
  const key = normalizeGenderKey(gender);
  if (!key) return null;
  return GENDER_LABEL[key] || null;
}

/**
 * Préfixe la civilité au nom affiché (ex. « Mme Matsa CHERIF »).
 * @param {string|null|undefined} name
 * @param {string|null|undefined} gender
 */
export function formatNameWithCivility(name, gender) {
  const trimmed = String(name || '').trim();
  if (!trimmed) return trimmed;
  const short = getGenderShortLabel(gender);
  return short ? `${short} ${trimmed}` : trimmed;
}

/**
 * Résout le genre passager depuis les champs API réservation.
 * @param {object|null|undefined} reservation
 * @param {object|null|undefined} identityView résultat buildIdentityFromApi
 */
export function resolvePassengerGender(reservation, identityView) {
  const fromIdentity = identityView?.passenger?.gender;
  if (fromIdentity) return fromIdentity;
  const fromPassenger = reservation?.passenger?.gender;
  if (fromPassenger) return fromPassenger;
  return reservation?.client?.gender || null;
}
