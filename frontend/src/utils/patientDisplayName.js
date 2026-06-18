/** Format unifié « NOM Prénom » — parité backend `format_patient_display_name_nom_prenom`. */

function capitalizeGivenNames(given) {
  const s = String(given || '').trim();
  if (!s) return s;
  return s
    .split(/\s+/)
    .map((part) =>
      part.includes('-')
        ? part
            .split('-')
            .map((seg) => (seg ? seg.charAt(0).toUpperCase() + seg.slice(1).toLowerCase() : seg))
            .join('-')
        : part.charAt(0).toUpperCase() + part.slice(1).toLowerCase()
    )
    .join(' ');
}

function formatFamilyNameToken(token) {
  const t = String(token || '').trim();
  if (!t) return t;
  if (t.includes('-')) {
    return t.split('-').map((seg) => seg.toUpperCase()).join('-');
  }
  return t.toUpperCase();
}

function isFamilyNameToken(token) {
  const t = String(token || '')
    .trim()
    .replace(/\.$/, '');
  if (!t || t.length < 2) return false;
  if (t.includes('-')) return t === t.toUpperCase();
  const letters = t.replace(/-/g, '').replace(/\./g, '');
  return letters.length >= 2 && t === t.toUpperCase();
}

export function formatPatientDisplayNameNomPrenom(raw) {
  if (raw == null) return '';
  const s = String(raw).trim().replace(/\s+/g, ' ');
  if (!s) return s;
  if (s.startsWith('Client #') || s === 'Client') return s;
  const parts = s.split(' ');
  if (parts.length === 1) return formatFamilyNameToken(parts[0]);

  if (isFamilyNameToken(parts[0])) {
    return `${formatFamilyNameToken(parts[0])} ${capitalizeGivenNames(parts.slice(1).join(' '))}`.trim();
  }
  if (isFamilyNameToken(parts[parts.length - 1])) {
    const nom = formatFamilyNameToken(parts[parts.length - 1]);
    const prenom = capitalizeGivenNames(parts.slice(0, -1).join(' '));
    return `${nom} ${prenom}`.trim();
  }
  const nom = formatFamilyNameToken(parts[parts.length - 1]);
  const prenom = capitalizeGivenNames(parts.slice(0, -1).join(' '));
  return `${nom} ${prenom}`.trim();
}
