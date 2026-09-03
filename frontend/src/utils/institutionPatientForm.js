/**
 * Normalisation / validation formulaire patient institution.
 * Aligné sur InstitutionPatientCreateSchema — PATIENT-IDENTITY-01.
 *
 * Mineur autorisé avec confirmation explicite (minor_dob_confirmed).
 */

import { getPhoneValidationError, normalizePhone } from './phone';

export const ISO_DATE_REGEX = /^\d{4}-\d{2}-\d{2}$/;
export const VALID_PATIENT_GENDERS = ['HOMME', 'FEMME', 'AUTRE'];
export const MIN_ADULT_AGE_YEARS = 18;
export const MINOR_DOB_CONFIRMATION_CODE = 'MINOR_DOB_CONFIRMATION_REQUIRED';

const FIELD_LABELS = {
  first_name: 'Prénom',
  last_name: 'Nom',
  dob: 'Date de naissance',
  gender: 'Civilité',
  phone: 'Téléphone',
  address: 'Adresse',
  city: 'Ville',
  postal_code: 'NPA',
  avs_number: 'N° AVS',
  insurance_name: 'Caisse maladie',
  insurance_number: 'N° assuré',
  guardian_phone: 'Téléphone curateur',
  guardian_email: 'Email curateur',
  guardianship_type: 'Type de curatelle',
};

function pad2(n) {
  return String(n).padStart(2, '0');
}

function startOfLocalDay(d = new Date()) {
  return new Date(d.getFullYear(), d.getMonth(), d.getDate());
}

/** Aujourd'hui en YYYY-MM-DD (local) — maxDate DOB. */
export function todayIso(today = new Date()) {
  const d = startOfLocalDay(today);
  return `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())}`;
}

/** DOB d'un patient ayant exactement 18 ans aujourd'hui. */
export function adultDobCutoff(today = new Date()) {
  const d = new Date(
    today.getFullYear() - MIN_ADULT_AGE_YEARS,
    today.getMonth(),
    today.getDate(),
  );
  return `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())}`;
}

export function parseLocalIsoDate(iso) {
  if (!ISO_DATE_REGEX.test(iso)) return null;
  const [y, m, d] = iso.split('-').map(Number);
  const dt = new Date(y, m - 1, d);
  if (
    dt.getFullYear() !== y
    || dt.getMonth() !== m - 1
    || dt.getDate() !== d
  ) {
    return null;
  }
  return dt;
}

export function patientAgeYears(isoDob, today = new Date()) {
  const dob = parseLocalIsoDate(isoDob);
  if (!dob) return null;
  const ref = startOfLocalDay(today);
  let years = ref.getFullYear() - dob.getFullYear();
  if (
    ref.getMonth() < dob.getMonth()
    || (ref.getMonth() === dob.getMonth() && ref.getDate() < dob.getDate())
  ) {
    years -= 1;
  }
  return years;
}

export function isMinorDob(isoDob, today = new Date()) {
  const age = patientAgeYears(isoDob, today);
  if (age === null) return false;
  return age < MIN_ADULT_AGE_YEARS;
}

/** Téléphone : chiffres et + initial uniquement. */
export function sanitizePhoneInput(raw) {
  const value = String(raw ?? '');
  const hasPlus = value.startsWith('+');
  const digits = value.replace(/\D/g, '');
  if (!digits) return hasPlus ? '+' : '';
  return hasPlus ? `+${digits}` : digits;
}

/** NPA suisse : chiffres uniquement (max 20 côté API). */
export function sanitizePostalCodeInput(raw) {
  const digits = String(raw ?? '').replace(/\D/g, '');
  if (!digits) return '';
  return digits.slice(0, 20);
}

/** AVS : chiffres et points, max 16 caractères. */
export function sanitizeAvsInput(raw) {
  const cleaned = String(raw ?? '').replace(/[^\d.]/g, '');
  if (!cleaned) return '';
  return cleaned.slice(0, 16);
}

/** Email : minuscules, sans espaces. */
export function sanitizeEmailInput(raw) {
  return String(raw ?? '').replace(/\s/g, '').toLowerCase();
}

/** Texte libre : trim + espaces multiples réduits. */
export function normalizeText(value, maxLen) {
  const text = String(value ?? '').replace(/\s+/g, ' ').trim();
  if (!text) return null;
  return maxLen ? text.slice(0, maxLen) : text;
}

/**
 * Valide une DOB : présente, calendrier réel, pas future.
 * Les mineurs sont ACCEPTÉS (confirmation séparée).
 */
export function normalizeDob(raw, { required = true } = {}) {
  const value = String(raw ?? '').trim();
  if (!value) {
    return {
      value: null,
      error: required ? 'Date de naissance requise' : null,
      isMinor: false,
      age: null,
    };
  }
  if (!ISO_DATE_REGEX.test(value)) {
    return {
      value: null,
      error: 'Date de naissance invalide (format attendu : AAAA-MM-JJ)',
      isMinor: false,
      age: null,
    };
  }
  const dob = parseLocalIsoDate(value);
  if (!dob) {
    return {
      value: null,
      error: 'Date de naissance invalide.',
      isMinor: false,
      age: null,
    };
  }
  const today = startOfLocalDay();
  if (dob > today) {
    return {
      value: null,
      error: 'La date de naissance ne peut pas être dans le futur',
      isMinor: false,
      age: null,
    };
  }
  const age = patientAgeYears(value, today);
  return {
    value,
    error: null,
    isMinor: age !== null && age < MIN_ADULT_AGE_YEARS,
    age,
  };
}

export function normalizeGender(raw, { required = true } = {}) {
  const value = String(raw ?? '').trim().toUpperCase();
  if (!value) {
    return {
      value: null,
      error: required ? 'Civilité requise' : null,
    };
  }
  if (VALID_PATIENT_GENDERS.includes(value)) return { value, error: null };
  return { value: null, error: 'Civilité invalide' };
}

export function normalizeEmail(raw) {
  const value = sanitizeEmailInput(raw);
  if (!value) return { value: null, error: null };
  if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value)) {
    return { value: null, error: 'Format email invalide' };
  }
  return { value: value.slice(0, 200), error: null };
}

/**
 * Prépare le payload API.
 * @param {object} options
 * @param {boolean} [options.forceCreate]
 * @param {boolean} [options.minorDobConfirmed] — confirmation UI mineur
 * @param {string|null} [options.previousDob] — DOB actuelle (édition) pour éviter
 *   de reconfirmer si inchangée
 */
export function buildInstitutionPatientPayload(
  formData,
  {
    forceCreate = false,
    minorDobConfirmed = false,
    previousDob = null,
  } = {},
) {
  const errors = [];

  const first_name = normalizeText(formData.first_name, 100);
  const last_nameRaw = normalizeText(formData.last_name, 100);
  const last_name = last_nameRaw ? last_nameRaw.toUpperCase() : null;
  if (!first_name) errors.push('Prénom requis');
  if (!last_name) errors.push('Nom requis');

  const phone = normalizePhone(formData.phone);
  const phoneError = getPhoneValidationError(phone);
  if (phoneError) errors.push(phoneError);

  const genderResult = normalizeGender(formData.gender, { required: true });
  if (genderResult.error) errors.push(genderResult.error);

  const dobResult = normalizeDob(formData.dob, { required: true });
  if (dobResult.error) errors.push(dobResult.error);

  const prevIso = previousDob ? String(previousDob).split('T')[0] : null;
  const dobChanged = !prevIso || prevIso !== dobResult.value;
  const needsMinorConfirmation = Boolean(
    dobResult.value && dobResult.isMinor && dobChanged,
  );

  const hasGuardianship = Boolean(formData.has_guardianship);

  let guardian_phone = null;
  let guardian_email = null;
  if (hasGuardianship) {
    guardian_phone = normalizePhone(formData.guardian_phone);
    const guardianPhoneError = getPhoneValidationError(guardian_phone);
    if (guardianPhoneError) errors.push(`Téléphone curateur : ${guardianPhoneError}`);

    const emailResult = normalizeEmail(formData.guardian_email);
    if (emailResult.error) errors.push(`Email curateur : ${emailResult.error}`);
    guardian_email = emailResult.value;
  }

  const postalDigits = sanitizePostalCodeInput(formData.postal_code);
  const address = normalizeText(formData.address, 255);
  const city = normalizeText(formData.city, 100);
  if (!address) errors.push('Adresse requise');
  if (!postalDigits) errors.push('NPA requis');
  if (!city) errors.push('Ville requise');

  const payload = {
    first_name,
    last_name,
    dob: dobResult.value,
    gender: genderResult.value,
    phone,
    address,
    city,
    postal_code: postalDigits || null,
    door_code: normalizeText(formData.door_code, 50),
    floor: normalizeText(formData.floor, 20),
    access_notes: normalizeText(formData.access_notes, 2000),
    residence_name: normalizeText(formData.residence_name, 200),
    avs_number: sanitizeAvsInput(formData.avs_number) || null,
    insurance_name: normalizeText(formData.insurance_name, 200),
    insurance_number: normalizeText(formData.insurance_number, 50),
    has_guardianship: hasGuardianship,
    notes: normalizeText(formData.notes, 2000),
  };

  if (hasGuardianship) {
    const guardianshipType = String(formData.guardianship_type || '').trim();
    payload.guardianship_type = guardianshipType || null;
    payload.guardian_name = normalizeText(formData.guardian_name, 200);
    payload.guardian_organization = normalizeText(formData.guardian_organization, 200);
    payload.guardian_phone = guardian_phone;
    payload.guardian_email = guardian_email;
    payload.guardian_address = normalizeText(formData.guardian_address, 500);
  } else {
    payload.guardianship_type = null;
    payload.guardian_name = null;
    payload.guardian_organization = null;
    payload.guardian_phone = null;
    payload.guardian_email = null;
    payload.guardian_address = null;
  }

  if (forceCreate) payload.force_create = true;
  if (needsMinorConfirmation && minorDobConfirmed) {
    payload.minor_dob_confirmed = true;
  }

  return {
    payload,
    errors,
    needsMinorConfirmation,
    minorAge: dobResult.age,
    isMinor: Boolean(dobResult.isMinor),
  };
}

/** Message lisible à partir d'une réponse API 400/422. */
export function formatInstitutionPatientApiError(data) {
  if (!data || typeof data !== 'object') return 'Erreur';

  if (data.code === MINOR_DOB_CONFIRMATION_CODE) {
    return data.error || 'Confirmation de la date de naissance mineure requise.';
  }

  const details = data.details;
  if (details && typeof details === 'object') {
    const parts = Object.entries(details).flatMap(([field, messages]) => {
      const list = Array.isArray(messages) ? messages : [messages];
      const label = FIELD_LABELS[field] || field;
      return list.filter(Boolean).map((msg) => `${label} : ${msg}`);
    });
    if (parts.length > 0) return parts.join(' · ');
  }

  if (typeof data.message === 'string' && data.message.trim()) return data.message;
  if (typeof data.error === 'string' && data.error.trim()) return data.error;
  return 'Erreur';
}

export function formatDobDisplay(iso) {
  const d = parseLocalIsoDate(iso);
  if (!d) return iso || '—';
  return `${pad2(d.getDate())}.${pad2(d.getMonth() + 1)}.${d.getFullYear()}`;
}

/** Normalise un objet formulaire chargé depuis l'API (édition). */
export function normalizePatientFormState(raw = {}) {
  return {
    gender: raw.gender || '',
    first_name: raw.first_name || '',
    last_name: raw.last_name || '',
    dob: raw.dob ? String(raw.dob).split('T')[0] : '',
    phone: sanitizePhoneInput(raw.phone || ''),
    address: raw.address || '',
    postal_code: sanitizePostalCodeInput(raw.postal_code || ''),
    city: raw.city || '',
    door_code: raw.door_code || '',
    floor: raw.floor || '',
    access_notes: raw.access_notes || '',
    residence_name: raw.residence_name || '',
    avs_number: sanitizeAvsInput(raw.avs_number || ''),
    insurance_name: raw.insurance_name || '',
    insurance_number: raw.insurance_number || '',
    has_guardianship: Boolean(raw.has_guardianship),
    guardianship_type: raw.guardianship_type || '',
    guardian_name: raw.guardian_name || '',
    guardian_organization: raw.guardian_organization || '',
    guardian_phone: sanitizePhoneInput(raw.guardian_phone || ''),
    guardian_email: sanitizeEmailInput(raw.guardian_email || ''),
    guardian_address: raw.guardian_address || '',
    notes: raw.notes || '',
  };
}
