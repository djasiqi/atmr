/**
 * Normalisation / validation formulaire patient institution.
 * Aligné sur InstitutionPatientCreateSchema (backend).
 */

import { getPhoneValidationError, normalizePhone } from './phone';

export const ISO_DATE_REGEX = /^\d{4}-\d{2}-\d{2}$/;
export const VALID_PATIENT_GENDERS = ['HOMME', 'FEMME', 'AUTRE'];

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

export function normalizeDob(raw) {
  const value = String(raw ?? '').trim();
  if (!value) return { value: null, error: null };
  if (ISO_DATE_REGEX.test(value)) return { value, error: null };
  return { value: null, error: 'Date de naissance invalide (format attendu : AAAA-MM-JJ)' };
}

export function normalizeGender(raw) {
  const value = String(raw ?? '').trim().toUpperCase();
  if (!value) return { value: null, error: null };
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
 * Prépare le payload API et collecte les erreurs bloquantes côté client.
 */
export function buildInstitutionPatientPayload(formData, { forceCreate = false } = {}) {
  const errors = [];

  const first_name = normalizeText(formData.first_name, 100);
  const last_nameRaw = normalizeText(formData.last_name, 100);
  const last_name = last_nameRaw ? last_nameRaw.toUpperCase() : null;
  if (!first_name) errors.push('Prénom requis');
  if (!last_name) errors.push('Nom requis');

  const phone = normalizePhone(formData.phone);
  const phoneError = getPhoneValidationError(phone);
  if (phoneError) errors.push(phoneError);

  const dobResult = normalizeDob(formData.dob);
  if (dobResult.error) errors.push(dobResult.error);

  const genderResult = normalizeGender(formData.gender);
  if (genderResult.error) errors.push(genderResult.error);

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

  const payload = {
    first_name,
    last_name,
    dob: dobResult.value,
    gender: genderResult.value,
    phone,
    address: normalizeText(formData.address, 255),
    city: normalizeText(formData.city, 100),
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

  return { payload, errors };
}

/** Message lisible à partir d'une réponse API 400. */
export function formatInstitutionPatientApiError(data) {
  if (!data || typeof data !== 'object') return 'Erreur';

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
