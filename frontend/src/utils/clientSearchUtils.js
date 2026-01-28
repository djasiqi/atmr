/**
 * Utilitaires de recherche client (page Gestion des clients).
 * Utilisés pour le filtre local côté frontend.
 *
 * - getClientDisplayName: nom affiché / clé de tri (affichage ↔ tri cohérents)
 * - normalizeText: lower, NFD, suppression diacritiques, collapse espaces
 * - buildSearchHaystack: concat des champs cherchables (first_name, last_name, etc.)
 * - clientMatchesSearch: match du query sur le haystack
 */

/**
 * Nom affiché d’un client. À utiliser pour l’affichage (ClientsTable) et le tri par nom
 * afin de garder cohérence affichage ↔ tri.
 *
 * @param {object} client
 * @returns {string}
 */
export function getClientDisplayName(client) {
  if (!client) return '';
  if (client.is_institution && client.institution_name) return client.institution_name;
  if (client.full_name && client.full_name !== 'Nom non renseigné') return client.full_name;
  const first = client.first_name ?? client.user_first_name ?? client.user?.first_name ?? '';
  const last = client.last_name ?? client.user_last_name ?? client.user?.last_name ?? '';
  const built = `${first} ${last}`.trim();
  return built || `Client #${client.id}`;
}

/**
 * Normalise un texte pour la recherche (casse, accents, espaces).
 * Tirets et apostrophes (-, ', ') → espace avant collapse, pour que
 * "el alaoui" / "el-alaoui" matchent "El-Alaoui", "d'artagnan" matche "D'Artagnan".
 * @param {string} text
 * @returns {string}
 */
export function normalizeText(text) {
  if (text == null || typeof text !== 'string') return '';
  return text
    .toLowerCase()
    .replace(/[-''\u2018\u2019]/g, ' ')
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/\s+/g, ' ')
    .trim();
}

/**
 * Construit une chaîne "haystack" à partir des champs cherchables du client.
 * Utilise les clés renvoyées par l’API (Client.serialize) : first_name, last_name,
 * full_name, user.username, institution_name, contact_email, phone, id, domicile,
 * birth_date (YYYY-MM-DD, dd/mm/yyyy, dd.mm.yyyy, ddmmyyyy), residence_facility, billing_address.
 *
 * @param {object} client - Objet client (liste ou détail)
 * @returns {string}
 */
export function buildSearchHaystack(client) {
  if (!client) return '';

  const first = client.first_name ?? client.user_first_name ?? client.user?.first_name ?? '';
  const last = client.last_name ?? client.user_last_name ?? client.user?.last_name ?? '';
  const full = client.full_name ?? `${first} ${last}`.trim();
  const username = client.username ?? client.user?.username ?? '';
  const institution = client.institution_name ?? '';
  const email = client.contact_email ?? '';
  const phone = client.contact_phone ?? client.phone ?? client.user?.phone ?? '';
  const id = String(client.id ?? '');
  const addr = client.domicile?.address ?? client.domicile_address ?? '';
  const zip = client.domicile?.zip ?? client.domicile_zip ?? '';
  const city = client.domicile?.city ?? client.domicile_city ?? '';
  const residence = client.residence_facility ?? '';
  const billing = client.billing_address ?? '';

  let birth = '';
  const bd = client.user_birth_date ?? client.user?.birth_date ?? client.birth_date;
  if (bd) {
    const s = typeof bd === 'string' ? bd : (bd.toISOString ? bd.toISOString().slice(0, 10) : '');
    birth = s;
    const [y, m, d] = s.split('-');
    if (y && m && d) {
      birth += ` ${d}/${m}/${y}`;
      birth += ` ${d}.${m}.${y}`;
      birth += ` ${d}${m}${y}`;
    }
  }

  const parts = [
    id,
    first,
    last,
    full,
    username,
    institution,
    email,
    phone,
    addr,
    zip,
    city,
    residence,
    billing,
    birth,
  ];
  return parts.filter(Boolean).join(' ');
}

/**
 * Indique si un client matche la requête de recherche.
 *
 * @param {object} client
 * @param {string} query
 * @returns {boolean}
 */
export function clientMatchesSearch(client, query) {
  const q = (query || '').trim();
  if (!q) return true;
  const haystack = buildSearchHaystack(client);
  const nq = normalizeText(q);
  const nh = normalizeText(haystack);
  return nh.includes(nq);
}

/**
 * Match rapide quand le haystack est déjà normalisé (pré-calculé).
 * À utiliser dans un filtre avec useMemo([clients]) pour éviter de rebuilder
 * les haystacks à chaque frappe.
 *
 * @param {string} normalizedHaystack - buildSearchHaystack(client) puis normalizeText(...)
 * @param {string} query
 * @returns {boolean}
 */
export function matchSearchQuery(normalizedHaystack, query) {
  const q = (query || '').trim();
  if (!q) return true;
  return normalizedHaystack.includes(normalizeText(q));
}
