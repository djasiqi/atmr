function cleanNamePart(value) {
  const text = String(value || '').trim();
  if (!text || text === 'Non spécifié') return '';
  return text;
}

/** Identifiant technique (osmani_mirjete_c8d0da), jamais un nom à afficher. */
export function looksLikeAccountHandle(value) {
  const text = String(value || '').trim();
  if (!text || text.includes(' ')) return false;
  return /_[0-9a-f]{4,}$/i.test(text);
}

export function fullNameFromUser(user) {
  const last = cleanNamePart(user?.last_name).toLocaleUpperCase('fr-CH');
  return `${cleanNamePart(user?.first_name)} ${last}`.trim();
}

/** Nom affiché du compte : prénom + nom, jamais l'identifiant technique. */
export function pickAccountDisplayName(userNameProp, sessionName) {
  const explicit = String(userNameProp || '').trim();
  if (
    explicit
    && explicit !== 'Utilisateur'
    && explicit !== 'Client'
    && !looksLikeAccountHandle(explicit)
  ) {
    return explicit;
  }
  const fromSession = String(sessionName || '').trim();
  if (fromSession && !looksLikeAccountHandle(fromSession)) return fromSession;
  return '';
}
