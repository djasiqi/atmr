/**
 * Utilitaires A/R (aperçu + édition brouillon facture).
 *
 * Deux notions distinctes, à ne jamais confondre :
 *
 * 1. INFORMATION — « la réservation a été créée en aller-retour » (badge éditeur « A/R »).
 *    Portée par des drapeaux (`billing_unit`, `transport_type`, `is_round_trip_leg`), parfois
 *    ajoutés dynamiquement par l'API à partir de `booking.is_round_trip`, même quand une seule
 *    jambe est facturée. Ne donne AUCUN droit fonctionnel et n'apparaît PAS sur la facture client.
 *    → `isSingleMergedRoundTripLine`, `isAnyRoundTripLine`, `lineEditorContextArTag`.
 *
 * 2. STRUCTURE — « cette ligne de facture contient effectivement aller + retour ».
 *    Déduite des réservations réellement rattachées (`booking_ids`, réservations secondaires,
 *    partenaire d'une paire deux lignes), jamais du prix ni des drapeaux ci-dessus.
 *    → `roundTripLineStructure`, `invoiceLineHasBothRoundTripLegs`,
 *      `canShowRoundTripLegExcludeActions`, `invoiceLineRepresentsFullRoundTrip`.
 *    Seule cette notion autorise « sans retour », « sans aller », « Retirer l'aller-retour
 *    complet » et le tag client `[A/R]` (aperçu HTML = PDF).
 *    Même règle que le backend `application.invoices.invoice_line_round_trip`.
 */

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

export function getInvoiceLineMeta(line) {
  return parseLineMeta(line?.line_meta);
}

/** Ligne masquée dans l’aperçu HTML (jambe retour d’une paire deux lignes). */
export function isRoundTripPreviewHiddenLine(line) {
  const m = getInvoiceLineMeta(line);
  return m?.preview_hide_merged_round_trip === true;
}

/** Ligne primaire A/R (aperçu : montants cumulés, partenaire masqué). */
export function isRoundTripPreviewPrimaryLine(line) {
  const m = getInvoiceLineMeta(line);
  if (m?.period_preview_single_leg) return false;
  return m?.round_trip_merge_partner_reservation_id != null;
}

/* ───────────────────────── STRUCTURE (droits fonctionnels) ───────────────────────── */

function addFiniteId(set, value) {
  if (value == null) return;
  const n = Number(value);
  if (Number.isFinite(n)) set.add(n);
}

/**
 * Réservations distinctes réellement rattachées à CETTE ligne (méta persistée).
 *
 * Même résolution que le backend `_split_single_merged_round_trip_line` : `booking_ids`
 * (≥ 2 entrées) sinon `reservation_id` + réservations secondaires. Le partenaire d'une paire
 * deux lignes (`round_trip_merge_partner_reservation_id`) n'est PAS compté : il vit sur une autre
 * ligne de la facture.
 */
export function attachedRoundTripBookingIds(line) {
  const m = getInvoiceLineMeta(line);
  if (!m) return [];
  const fromBookingIds = new Set();
  if (Array.isArray(m.booking_ids)) {
    m.booking_ids.forEach((id) => addFiniteId(fromBookingIds, id));
  }
  if (fromBookingIds.size >= 2) return [...fromBookingIds];
  if (line?.reservation_id == null) return [...fromBookingIds];
  const withSecondary = new Set();
  addFiniteId(withSecondary, line.reservation_id);
  const sec = m.round_trip_secondary_reservation_ids;
  if (Array.isArray(sec)) {
    sec.forEach((id) => addFiniteId(withSecondary, id));
  } else {
    addFiniteId(withSecondary, m.round_trip_secondary_reservation_id);
  }
  return withSecondary.size >= 2 ? [...withSecondary] : [...fromBookingIds];
}

/**
 * Cette ligne contient effectivement aller + retour (≥ 2 réservations rattachées).
 *
 * Ne se déduit jamais du prix ni de `billing_unit` / `transport_type` : l'API ajoute ces drapeaux
 * aux lignes dont la réservation est `is_round_trip` même quand une seule jambe est facturée.
 * Faux après exclusion d'une jambe dans l'aperçu période (`period_preview_single_leg`).
 */
export function invoiceLineHasBothRoundTripLegs(line) {
  const m = getInvoiceLineMeta(line);
  if (!m || m.period_preview_single_leg) return false;
  return attachedRoundTripBookingIds(line).length >= 2;
}

export const ROUND_TRIP_LINE_STRUCTURE = Object.freeze({
  /** Aller + retour rattachés à cette ligne : découpage possible, corbeille = retrait complet. */
  MERGED_BOTH_LEGS: 'merged_both_legs',
  /** Aller d'une paire deux lignes (retour = autre ligne) : découpage possible, corbeille = l'aller. */
  PAIR_PRIMARY: 'pair_primary',
  /** Retour d'une paire deux lignes : corbeille = le retour, pas de découpage. */
  PAIR_RETURN: 'pair_return',
  /** Une seule réservation (même si créée en A/R) : aucune action de découpage. */
  SINGLE: 'single',
});

/**
 * Nature structurelle d'une ligne vis-à-vis de l'aller-retour. Pilote toutes les actions
 * (« sans retour », « sans aller », libellé / confirmation de la corbeille).
 *
 * @param {object} line Ligne facture (brouillon ou aperçu).
 * @param {object[]} [allLines] Lignes de la même facture : si fourni, une paire deux lignes n'est
 *   reconnue que si l'autre jambe est réellement présente (méta de paire obsolète ⇒ `SINGLE`).
 */
export function roundTripLineStructure(line, allLines) {
  const m = getInvoiceLineMeta(line);
  if (!m) return ROUND_TRIP_LINE_STRUCTURE.SINGLE;
  if (m.period_preview_single_leg) return ROUND_TRIP_LINE_STRUCTURE.SINGLE;
  if (invoiceLineHasBothRoundTripLegs(line)) return ROUND_TRIP_LINE_STRUCTURE.MERGED_BOTH_LEGS;
  const partnerPresent = allLines === undefined || getRoundTripPartnerLine(line, allLines) != null;
  if (m.round_trip_merge_partner_reservation_id != null) {
    return partnerPresent
      ? ROUND_TRIP_LINE_STRUCTURE.PAIR_PRIMARY
      : ROUND_TRIP_LINE_STRUCTURE.SINGLE;
  }
  if (m.preview_hide_merged_round_trip === true) {
    return partnerPresent
      ? ROUND_TRIP_LINE_STRUCTURE.PAIR_RETURN
      : ROUND_TRIP_LINE_STRUCTURE.SINGLE;
  }
  return ROUND_TRIP_LINE_STRUCTURE.SINGLE;
}

/**
 * Afficher les liens « sans retour » / « sans aller » (aperçu période + brouillon).
 * Uniquement si les deux jambes existent réellement : sur la ligne (fusion) ou en paire deux lignes.
 */
export function canShowRoundTripLegExcludeActions(line, allLines) {
  const s = roundTripLineStructure(line, allLines);
  return (
    s === ROUND_TRIP_LINE_STRUCTURE.MERGED_BOTH_LEGS || s === ROUND_TRIP_LINE_STRUCTURE.PAIR_PRIMARY
  );
}

/**
 * Tag client `[A/R]` (aperçu HTML = PDF) : la ligne facture réellement un aller-retour.
 *
 * Préfère le champ API `invoice_line_represents_full_round_trip` s'il est booléen
 * (source de vérité backend). Sinon retombe sur `roundTripLineStructure`.
 * Ne lit jamais le prix ni `billing_unit` / `transport_type`.
 */
export function invoiceLineRepresentsFullRoundTrip(line, allLines) {
  if (typeof line?.invoice_line_represents_full_round_trip === 'boolean') {
    return line.invoice_line_represents_full_round_trip;
  }
  const s = roundTripLineStructure(line, allLines);
  return (
    s === ROUND_TRIP_LINE_STRUCTURE.MERGED_BOTH_LEGS || s === ROUND_TRIP_LINE_STRUCTURE.PAIR_PRIMARY
  );
}

/** Étiquette facture client : `A/R` ou rien. */
export function invoiceLineClientArTag(line, allLines) {
  return invoiceLineRepresentsFullRoundTrip(line, allLines) ? 'A/R' : null;
}

/* ───────────────────────── INFORMATION (badge « A/R » uniquement) ───────────────────────── */

/**
 * Ligne A/R facturée en une seule entrée facture (génération S1 / S2, ou réservation `is_round_trip`).
 *
 * INFORMATION uniquement (badge) : `billing_unit === 'round_trip'` est aussi posé par l'API quand
 * une seule jambe est facturée. Pour toute décision (découpage, retrait complet), utiliser
 * `roundTripLineStructure` / `invoiceLineHasBothRoundTripLegs`.
 */
export function isSingleMergedRoundTripLine(line) {
  const m = getInvoiceLineMeta(line);
  if (!m) return false;
  if (m.billing_unit === 'round_trip') return true;
  const sec = m.round_trip_secondary_reservation_ids;
  if (Array.isArray(sec) && sec.length > 0) return true;
  return m.round_trip_secondary_reservation_id != null;
}

/** INFORMATION uniquement (badge / libellés) : la ligne est liée à un aller-retour, d'une façon ou d'une autre. */
export function isAnyRoundTripLine(line) {
  return (
    isRoundTripPreviewPrimaryLine(line) ||
    isRoundTripPreviewHiddenLine(line) ||
    isSingleMergedRoundTripLine(line)
  );
}

export function findInvoiceLineByReservationId(lines, reservationId) {
  if (reservationId == null || !Number.isFinite(Number(reservationId))) return null;
  const rid = Number(reservationId);
  const list = Array.isArray(lines) ? lines : [];
  return list.find((ln) => Number(ln?.reservation_id) === rid) ?? null;
}

/** Partenaire A/R (paire deux lignes). */
export function getRoundTripPartnerLine(line, allLines) {
  const m = getInvoiceLineMeta(line);
  if (!m) return null;
  if (m.round_trip_merge_partner_reservation_id != null) {
    return findInvoiceLineByReservationId(allLines, m.round_trip_merge_partner_reservation_id);
  }
  if (m.round_trip_merge_primary_reservation_id != null) {
    return findInvoiceLineByReservationId(allLines, m.round_trip_merge_primary_reservation_id);
  }
  return null;
}

export function lineServiceDateSortKey(line) {
  const meta = getInvoiceLineMeta(line);
  const raw = meta?.service_date ?? meta?.service_date_iso;
  if (raw == null || String(raw).trim() === '') return '9999-12-31';
  const s = String(raw).trim();
  const dm = /^(\d{4})-(\d{2})-(\d{2})/.exec(s);
  if (dm) return `${dm[1]}-${dm[2]}-${dm[3]}`;
  const d = new Date(s);
  if (!Number.isNaN(d.getTime())) {
    return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
  }
  return '9999-12-31';
}

function invoiceLineEditorSortRank(line) {
  const m = getInvoiceLineMeta(line);
  if (m?.global_discount_line || m?.per_line_discount_line) return 2;
  const t = String(line?.type ?? line?.line_type ?? '')
    .trim()
    .toLowerCase();
  if (t === 'ride' || t === 'material_delivery') return 0;
  return 1;
}

function linePatientSortKey(line) {
  const meta = getInvoiceLineMeta(line);
  return String(meta?.patient_name ?? '')
    .trim()
    .toLowerCase();
}

/** Tri chronologique pour l’éditeur (parité aperçu / PDF S2). */
export function sortInvoiceLinesForEditor(lines) {
  const raw = Array.isArray(lines)
    ? lines.filter((ln) => ln != null && typeof ln === 'object')
    : [];
  return raw
    .map((ln, idx) => ({ ln, idx }))
    .sort((a, b) => {
      const ra = invoiceLineEditorSortRank(a.ln);
      const rb = invoiceLineEditorSortRank(b.ln);
      if (ra !== rb) return ra - rb;
      const da = lineServiceDateSortKey(a.ln);
      const db = lineServiceDateSortKey(b.ln);
      if (da !== db) return da.localeCompare(db);
      const pa = linePatientSortKey(a.ln);
      const pb = linePatientSortKey(b.ln);
      if (pa !== pb) return pa.localeCompare(pb);
      return a.idx - b.idx;
    })
    .map(({ ln }) => ln);
}

export function formatServiceDateFr(raw) {
  if (raw == null || raw === '') return null;
  const s = String(raw).trim();
  const dm = /^(\d{4})-(\d{2})-(\d{2})/.exec(s);
  if (dm) return `${dm[3]}.${dm[2]}.${dm[1]}`;
  const d = new Date(s);
  if (!Number.isNaN(d.getTime())) {
    return `${String(d.getDate()).padStart(2, '0')}.${String(d.getMonth() + 1).padStart(2, '0')}.${d.getFullYear()}`;
  }
  return s.slice(0, 10);
}

export function lineEditorContextSubline(line) {
  const meta = getInvoiceLineMeta(line);
  if (!meta) return null;
  const parts = [];
  if (meta.patient_name) parts.push(`Client : ${String(meta.patient_name).trim()}`);
  const dateRaw = meta.service_date ?? meta.service_date_iso;
  const dateLbl = formatServiceDateFr(dateRaw);
  if (dateLbl) parts.push(dateLbl);
  return parts.length ? parts.join(' · ') : null;
}

function normalizeRouteEndpoint(value) {
  return String(value || '')
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .toLowerCase()
    .replace(/[^\w\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function splitTransportEndpoints(description) {
  const raw = String(description || '')
    .trim()
    .replace(/^trajet\s*:\s*/i, '')
    .replace(/^trajet\s+/i, '');
  const parts = raw.split(/\s*(?:→|↔|<->)\s*/);
  if (parts.length < 2) return null;
  const start = normalizeRouteEndpoint(parts[0]);
  const end = normalizeRouteEndpoint(parts[parts.length - 1]);
  if (!start || !end) return null;
  return [start, end];
}

/** Vrai seulement pour A→B et B→A. Une chaîne A→B, B→C, C→A n'est pas un aller-retour. */
export function transportDescriptionsAreStrictReverse(left, right) {
  const a = splitTransportEndpoints(left);
  const b = splitTransportEndpoints(right);
  if (!a || !b) return false;
  return a[0] === b[1] && a[1] === b[0];
}

/** Inverse « Trajet A → B » en « Trajet B → A » (jambe retour si description partenaire absente). */
export function invertTrajetLineDescription(description) {
  if (description == null) return description ?? '';
  const s = String(description).trim();
  const m = /^Trajet\s+(.+?)\s*→\s*(.+)$/su.exec(s);
  if (!m) return s;
  return `Trajet ${m[2].trim()} → ${m[1].trim()}`;
}

/** Étiquette A/R inline dans la ligne contexte (sans badge séparé). INFORMATION uniquement. */
export function lineEditorContextArTag(line) {
  const m = getInvoiceLineMeta(line);
  if (m?.period_preview_single_leg) return null;
  if (isRoundTripPreviewHiddenLine(line)) return 'Retour';
  if (isRoundTripPreviewPrimaryLine(line)) return 'A/R';
  if (isSingleMergedRoundTripLine(line)) return 'A/R';
  return null;
}

/** Libellé de jambe (INFORMATION uniquement). */
export function roundTripLegLabel(line) {
  if (isRoundTripPreviewHiddenLine(line)) return 'Retour';
  if (isRoundTripPreviewPrimaryLine(line)) return 'Aller';
  if (isSingleMergedRoundTripLine(line)) return 'A/R';
  return null;
}

/**
 * Jambes A/R pour l'interface de contrôle (dépliable).
 * Le PDF reste compact ; ici on expose les booking_id + montants.
 * Basé sur la STRUCTURE : une ligne mono-réservation taguée A/R ne revendique pas « 2 courses ».
 */
export function getRoundTripAuditLegs(line) {
  const m = getInvoiceLineMeta(line);
  if (!m || typeof m !== 'object') return null;
  const structure = roundTripLineStructure(line);
  if (
    structure !== ROUND_TRIP_LINE_STRUCTURE.MERGED_BOTH_LEGS &&
    structure !== ROUND_TRIP_LINE_STRUCTURE.PAIR_PRIMARY
  ) {
    return null;
  }
  const primaryId = Number(m.primary_booking_id ?? line?.reservation_id ?? line?.booking_id);
  const partnerId = Number(
    m.round_trip_merge_partner_reservation_id ??
      m.round_trip_secondary_reservation_id ??
      (Array.isArray(m.round_trip_secondary_reservation_ids)
        ? m.round_trip_secondary_reservation_ids[0]
        : null)
  );
  const bookingIds = Array.isArray(m.booking_ids)
    ? m.booking_ids.map((id) => Number(id)).filter((id) => Number.isFinite(id))
    : [];
  const outboundId = Number.isFinite(primaryId) ? primaryId : bookingIds[0];
  const returnId = Number.isFinite(partnerId)
    ? partnerId
    : bookingIds.find((id) => id !== outboundId);
  if (!Number.isFinite(outboundId) && !Number.isFinite(returnId)) return null;
  const primaryHt = Number(m.round_trip_primary_amount_ht);
  const partnerHt = Number(m.round_trip_partner_amount_ht);
  return {
    segmentsCount: bookingIds.length >= 2 ? bookingIds.length : 2,
    outbound: {
      bookingId: Number.isFinite(outboundId) ? outboundId : null,
      amountHt: Number.isFinite(primaryHt) ? primaryHt : null,
      description:
        m.round_trip_primary_description != null
          ? String(m.round_trip_primary_description).trim()
          : null,
    },
    inbound: {
      bookingId: Number.isFinite(returnId) ? returnId : null,
      amountHt: Number.isFinite(partnerHt) ? partnerHt : null,
      description:
        m.round_trip_partner_description != null
          ? String(m.round_trip_partner_description).trim()
          : null,
    },
  };
}
