/** Présentation des lignes de facture (aperçu période) — aucune logique financière. */

export const formatPreviewDayMonth = (iso) => {
  const s = String(iso || '').trim();
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(s);
  if (m) return `${m[3]}.${m[2]}`;
  const d = new Date(s);
  if (!Number.isNaN(d.getTime())) {
    return `${String(d.getDate()).padStart(2, '0')}.${String(d.getMonth() + 1).padStart(2, '0')}`;
  }
  return '';
};

export const compactInvoiceRoute = (description) => {
  const s = String(description || '').trim();
  if (!s) return '';
  const trajet = /^Trajet(?:\s*:\s*|\s+)(.+)$/iu.exec(s);
  if (trajet) return trajet[1].trim();
  return s;
};

const toFiniteNumber = (value) => {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
};

export const presentInvoiceLinesPreview = (previewLines = [], options = {}) => {
  const lines = Array.isArray(previewLines)
    ? previewLines.filter((line) => line && line.already_invoiced !== true)
    : [];
  const rows = lines.map((line) => {
    const bookingIds = Array.isArray(line.booking_ids)
      ? line.booking_ids.map((id) => Number(id)).filter((id) => Number.isFinite(id))
      : [Number(line.booking_id)].filter((id) => Number.isFinite(id));
    const partnerId = toFiniteNumber(line.round_trip_partner_booking_id);
    const segmentsCount = Number(line.segments_count) || bookingIds.length || 1;
    const isRoundTrip =
      segmentsCount >= 2 &&
      (line.unit_type === 'round_trip' ||
        Boolean(line.is_round_trip_leg && partnerId != null));
    return {
      key: line.preview_row_id || `booking:${line.booking_id}`,
      dateLabel: formatPreviewDayMonth(line.scheduled_at),
      patientName: String(line.patient_name || '').trim(),
      route: compactInvoiceRoute(line.description),
      amountHt: Number(line.amount_ht) || 0,
      isRoundTrip,
      segmentsCount,
      outboundBookingId: toFiniteNumber(line.booking_id) ?? bookingIds[0] ?? null,
      returnBookingId:
        toFiniteNumber(line.round_trip_partner_booking_id) ?? bookingIds[1] ?? null,
      outboundAmountHt: toFiniteNumber(line.round_trip_primary_amount_ht),
      returnAmountHt: toFiniteNumber(line.round_trip_partner_amount_ht),
      outboundDescription: compactInvoiceRoute(line.description),
      returnDescription: compactInvoiceRoute(line.round_trip_partner_description),
      legLabel: isRoundTrip ? '' : String(line.leg_label || '').trim(),
    };
  });
  const visualLineCount = rows.length;
  const prestationCountFromLines = rows.reduce(
    (n, row) => n + (Number(row.segmentsCount) || 1),
    0
  );
  const prestationCount = Number(options.prestationCount) || prestationCountFromLines;
  const totalHt =
    options.totalHt != null && Number.isFinite(Number(options.totalHt))
      ? Number(options.totalHt)
      : rows.reduce((n, row) => n + row.amountHt, 0);
  return {
    visualLineCount,
    prestationCount,
    totalHt,
    rows,
  };
};

/** Affiche les montants déjà calculés. Ne soustrait rien du tarif. */
export function presentCashPreview(preview) {
  if (!preview || preview.collected_chf == null || preview.remaining_due_chf == null) {
    return null;
  }
  if (preview.estimated_subtotal_ht == null) return null;
  const lines = Array.isArray(preview.preview_lines) ? preview.preview_lines : [];
  const legs = lines.flatMap((line) => {
    const source = Array.isArray(line.cash_legs) && line.cash_legs.length ? line.cash_legs : [line];
    return source.flatMap((leg) => {
      if (leg.recognized_net_chf == null || leg.remaining_due_chf == null) return [];
      return [
        {
          bookingId: leg.booking_id ?? line.booking_id,
          contractual: formatChfLabel(leg.contractual_chf ?? leg.amount_ht),
          collected: formatChfLabel(leg.recognized_net_chf),
          remaining: formatChfLabel(leg.remaining_due_chf),
        },
      ];
    });
  });
  return {
    services: formatChfLabel(preview.estimated_subtotal_ht),
    collected: formatChfLabel(preview.collected_chf),
    remaining: formatChfLabel(preview.remaining_due_chf),
    legs,
  };
}

function formatChfLabel(value) {
  if (value == null || value === '') return null;
  const raw = String(value).trim();
  if (!raw) return null;
  if (raw.includes('CHF')) return raw;
  const amount = Number(raw);
  if (!Number.isFinite(amount)) return null;
  return `${amount.toFixed(2)} CHF`;
}
