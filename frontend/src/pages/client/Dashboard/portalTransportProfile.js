/** Profil = habitudes. La réservation en fait une copie, jamais l'inverse. */

export function portalTransportProfileGaps(profile) {
  if (!profile) return ['profile'];
  const gaps = [];
  const first = String(profile.first_name || profile.user?.first_name || '').trim();
  const last = String(profile.last_name || profile.user?.last_name || '').trim();
  const birth = String(profile.birth_date || profile.user?.birth_date || '').trim();
  const phone = String(profile.phone || profile.user?.phone || profile.contact_phone || '').trim();
  const address = String(
    profile.domicile?.address || profile.user?.address || profile.address || profile.domicile_address || ''
  ).trim();
  const lat = profile.domicile?.lat ?? profile.domicile_lat;
  if (!first) gaps.push('first_name');
  if (!last) gaps.push('last_name');
  if (!birth) gaps.push('birth_date');
  if (profile.phone_verified !== true || !phone) gaps.push('phone');
  if (!address) gaps.push('address');
  else if (lat == null || lat === '') gaps.push('address_validated');
  return gaps;
}

/** Notes d'accès habituelles, prêtes à être copiées dans la course. */
export function composeProfilePickupAccess(profile) {
  const access = profile?.access || {};
  const floor = String(access.floor || profile?.floor || '').trim();
  const door = String(access.door_code || profile?.door_code || '').trim();
  const notes = String(access.notes || profile?.access_notes || '').trim();
  const lines = [];
  if (floor) {
    const lower = floor.toLowerCase();
    lines.push(lower.includes('étage') || lower.includes('etage') ? floor : `${floor} étage`);
  }
  if (door) lines.push(door.toLowerCase().startsWith('interphone') ? door : `Interphone ${door}`);
  if (notes) lines.push(notes);
  return lines.join('\n');
}

const MEDICAL_STEMS = [
  'hopital',
  'hospital',
  'clinique',
  'clinic',
  'cabinet',
  'docteur',
  'medecin',
  'medical',
  'medicale',
  'medicaux',
  'dentiste',
  'dentaire',
  'orthodont',
  'ophtalmolog',
  'chirurg',
  'polyclinique',
  'maternite',
  'urgences',
  'radiolog',
  'cardiolog',
  'oncolog',
  'pediatr',
  'gynecolog',
  'dermatolog',
  'neurolog',
  'orthoped',
  'urolog',
  'gastroenterolog',
  'pneumolog',
  'psychiatr',
  'dialyse',
  'laboratoire',
  'pharmacie',
  'kinesitherap',
  'physiotherap',
  'osteopath',
  'podologue',
  'podologie',
];

/** Sigles courts : uniquement en mot entier, pour éviter « Orléans » ou « endroit ». */
const MEDICAL_TOKENS = ['dr', 'ems', 'ehpad', 'cms', 'hug', 'chuv', 'orl', 'irm'];

function foldMedicalText(text) {
  return String(text || '')
    .toLowerCase()
    .normalize('NFD')
    .replace(/\p{M}/gu, '');
}

/** Un libellé de médecin (Dr, Dr méd., Docteur) n’est pas un nom d’établissement. */
const DOCTOR_HEADING = /^(?:dr(?:\s+med)?|docteur|medecin)\b/;

export const PORTAL_DOCTOR_FACILITY_LABEL = 'Cabinet médical';

function primaryPlaceLabel(text) {
  const raw = String(text || '').trim();
  const first = raw
    .split(',')
    .map((part) => part.trim())
    .find(Boolean);
  return first || raw;
}

export function classifyPortalMedicalPlace(text) {
  const label = primaryPlaceLabel(text);
  const folded = foldMedicalText(label).replace(/\./g, ' ').replace(/\s+/g, ' ').trim();
  if (DOCTOR_HEADING.test(folded)) {
    return { facility: PORTAL_DOCTOR_FACILITY_LABEL, doctor: label };
  }
  return { facility: label, doctor: '' };
}

/** Vrai si l’adresse ressemble à un lieu de soins (hôpital, clinique, cabinet, spécialité…). */
export function destinationLooksMedical(text) {
  const folded = foldMedicalText(text);
  if (!folded.trim()) return false;
  if (MEDICAL_STEMS.some((stem) => folded.includes(stem))) return true;
  return MEDICAL_TOKENS.some((token) =>
    new RegExp(`(?:^|[^a-z0-9])${token}(?![a-z0-9])`).test(folded)
  );
}
