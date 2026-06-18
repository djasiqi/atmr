const MONTHS_FR = [
  'janvier',
  'fevrier',
  'mars',
  'avril',
  'mai',
  'juin',
  'juillet',
  'aout',
  'septembre',
  'octobre',
  'novembre',
  'decembre',
];

/** Supprime les accents et met en minuscules pour recherche tolérante. */
export function normalizeInvoiceLineSearchText(value) {
  if (value == null || value === '') return '';
  return String(value)
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .toLowerCase()
    .trim();
}

function parseLineMeta(raw) {
  if (raw == null) return null;
  if (typeof raw === 'string') {
    try {
      const p = JSON.parse(raw);
      return typeof p === 'object' && p !== null ? p : null;
    } catch {
      return null;
    }
  }
  if (typeof raw === 'object') return raw;
  return null;
}

function pushUnique(parts, value) {
  if (value == null || value === '') return;
  const s = String(value).trim();
  if (s) parts.push(s);
}

/** Variantes de date affichables / saisissables pour une date ISO `YYYY-MM-DD`. */
export function serviceDateIsoToSearchVariants(iso) {
  if (iso == null || iso === '') return [];
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(String(iso).trim());
  if (!m) return [String(iso).trim()];
  const [, y, mo, mm] = m;
  const d = parseInt(mm, 10);
  const monthIdx = parseInt(mo, 10) - 1;
  const monthName = MONTHS_FR[monthIdx] ?? '';
  return [
    `${y}-${mo}-${mm}`,
    `${d}.${mo}.${y}`,
    `${String(d).padStart(2, '0')}.${mo}.${y}`,
    `${d}/${mo}/${y}`,
    `${String(d).padStart(2, '0')}/${mo}/${y}`,
    `${d}-${mo}-${y}`,
    `${mo}.${y}`,
    `${mo}/${y}`,
    `${mo}.${d}`,
    `${d}.${mo}`,
    `${y}-${mo}`,
    monthName,
    monthName ? `${monthName} ${y}` : '',
    `${d}${mo}${y}`,
    `${String(d).padStart(2, '0')}${mo}${y}`,
  ].filter(Boolean);
}

/** Nom patient : ordre « nom prénom » et « prénom nom » + tokens isolés. */
export function patientNameToSearchVariants(name) {
  if (name == null || String(name).trim() === '') return [];
  const raw = String(name).trim();
  const parts = [raw, raw.replace(/-/g, ' ')];
  const tokens = raw.split(/\s+/).filter(Boolean);
  if (tokens.length >= 2) {
    parts.push(tokens.join(' '));
    parts.push([...tokens].reverse().join(' '));
    parts.push(...tokens);
  }
  return parts;
}

/**
 * Chaîne de recherche normalisée regroupant libellé, client, date, montant, n° ligne, réservation, note.
 */
export function buildInvoiceLineFilterHaystack(line) {
  if (!line || typeof line !== 'object') return '';

  const chunks = [];

  pushUnique(chunks, line.description);
  pushUnique(chunks, line.type);
  pushUnique(chunks, line.line_type);
  pushUnique(chunks, line.id);
  pushUnique(chunks, line.reservation_id);
  pushUnique(chunks, line.adjustment_note);

  const ht = Number(line.line_total);
  if (Number.isFinite(ht)) {
    pushUnique(chunks, ht.toFixed(2));
    pushUnique(chunks, String(ht));
  }

  const meta = parseLineMeta(line.line_meta);
  if (meta) {
    pushUnique(chunks, meta.patient_name);
    for (const v of patientNameToSearchVariants(meta.patient_name)) {
      pushUnique(chunks, v);
    }

    const dateRaw = meta.service_date ?? meta.service_date_iso;
    for (const v of serviceDateIsoToSearchVariants(dateRaw)) {
      pushUnique(chunks, v);
    }

    const dateEndRaw = meta.service_date_end ?? meta.service_date_iso_end;
    for (const v of serviceDateIsoToSearchVariants(dateEndRaw)) {
      pushUnique(chunks, v);
    }

    const cp = meta.custom_prestation;
    if (cp && typeof cp === 'object') {
      pushUnique(chunks, cp.mode);
      pushUnique(chunks, cp.time_unit);
    }
  }

  return normalizeInvoiceLineSearchText(chunks.join(' '));
}

/** Découpe la requête en tokens (espaces) ; conserve dates et noms composés tels quels. */
export function tokenizeInvoiceLineFilterQuery(rawQuery) {
  const q = normalizeInvoiceLineSearchText(rawQuery);
  if (!q) return [];
  return q.split(/\s+/).filter(Boolean);
}

/**
 * Filtre professionnel : chaque mot de la requête doit correspondre (client prénom/nom dans les deux sens, date, libellé, montant, n°).
 */
export function invoiceLineMatchesFilter(line, rawQuery) {
  const tokens = tokenizeInvoiceLineFilterQuery(rawQuery);
  if (tokens.length === 0) return true;

  const haystack = buildInvoiceLineFilterHaystack(line);
  if (!haystack) return false;

  return tokens.every((token) => haystack.includes(token));
}

/** Filtre une liste de lignes ; conserve l’ordre d’origine. */
export function filterInvoiceLines(lines, rawQuery) {
  const list = Array.isArray(lines) ? lines : [];
  const tokens = tokenizeInvoiceLineFilterQuery(rawQuery);
  if (tokens.length === 0) return list;
  return list.filter((line) => invoiceLineMatchesFilter(line, rawQuery));
}
