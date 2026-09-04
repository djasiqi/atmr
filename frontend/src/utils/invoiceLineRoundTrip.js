/** Utilitaires A/R (aperçu + édition brouillon facture). */

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

/** Afficher les liens « sans retour » / « sans aller » (aperçu période + brouillon). */
export function canShowRoundTripLegExcludeActions(line) {
  const m = getInvoiceLineMeta(line);
  if (m?.period_preview_single_leg) return false;
  if (isRoundTripPreviewPrimaryLine(line)) return true;
  if (isSingleMergedRoundTripLine(line)) return true;
  return false;
}

/** Ligne A/R fusionnée en une seule entrée facture (génération S1). */
export function isSingleMergedRoundTripLine(line) {
  const m = getInvoiceLineMeta(line);
  if (!m) return false;
  if (m.billing_unit === 'round_trip') return true;
  const sec = m.round_trip_secondary_reservation_ids;
  if (Array.isArray(sec) && sec.length > 0) return true;
  return m.round_trip_secondary_reservation_id != null;
}

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
    return findInvoiceLineByReservationId(
      allLines,
      m.round_trip_merge_partner_reservation_id
    );
  }
  if (m.round_trip_merge_primary_reservation_id != null) {
    return findInvoiceLineByReservationId(
      allLines,
      m.round_trip_merge_primary_reservation_id
    );
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
  const raw = Array.isArray(lines) ? lines.filter((ln) => ln != null && typeof ln === 'object') : [];
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

/** Inverse « Trajet A → B » en « Trajet B → A » (jambe retour si description partenaire absente). */
export function invertTrajetLineDescription(description) {
  if (description == null) return description ?? '';
  const s = String(description).trim();
  const m = /^Trajet\s+(.+?)\s*→\s*(.+)$/su.exec(s);
  if (!m) return s;
  return `Trajet ${m[2].trim()} → ${m[1].trim()}`;
}

/** Étiquette A/R inline dans la ligne contexte (sans badge séparé). */
export function lineEditorContextArTag(line) {
  const m = getInvoiceLineMeta(line);
  if (m?.period_preview_single_leg) return null;
  if (isRoundTripPreviewHiddenLine(line)) return 'Retour';
  if (isRoundTripPreviewPrimaryLine(line)) return 'A/R';
  if (isSingleMergedRoundTripLine(line)) return 'A/R';
  return null;
}

export function roundTripLegLabel(line) {
  if (isRoundTripPreviewHiddenLine(line)) return 'Retour';
  if (isRoundTripPreviewPrimaryLine(line)) return 'Aller';
  if (isSingleMergedRoundTripLine(line)) return 'A/R';
  return null;
}

/**
 * Jambes A/R pour l'interface de contrôle (dépliable).
 * Le PDF reste compact ; ici on expose les booking_id + montants.
 */
export function getRoundTripAuditLegs(line) {
  const m = getInvoiceLineMeta(line);
  if (!m || typeof m !== 'object') return null;
  if (m.period_preview_single_leg) return null;
  const primaryId = Number(
    m.primary_booking_id ?? line?.reservation_id ?? line?.booking_id
  );
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
  const isAr =
    isRoundTripPreviewPrimaryLine(line) || isSingleMergedRoundTripLine(line);
  if (!isAr) return null;
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
