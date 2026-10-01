/**
 * Note de ligne facture visible par le client — parité HTML / PDF.
 *
 * Champ canonique : `adjustment_note` (déjà affiché par InvoiceLivePreview).
 * Ne pas inventer `customer_visible_note`. Ne jamais lire notes médicales,
 * motifs internes d'annulation, ni notes de niveau facture.
 */

export const CANONICAL_CLIENT_VISIBLE_FIELD = 'adjustment_note';

export function normalizeCustomerVisibleNote(raw) {
  if (raw == null) return null;
  const text = String(raw).trim();
  return text === '' ? null : text;
}

export function collectCustomerVisibleNotes(lines) {
  const notes = [];
  const seen = new Set();
  for (const line of Array.isArray(lines) ? lines : []) {
    const note = normalizeCustomerVisibleNote(
      line && typeof line === 'object' ? line[CANONICAL_CLIENT_VISIBLE_FIELD] : null,
    );
    if (note == null || seen.has(note)) continue;
    seen.add(note);
    notes.push(note);
  }
  return notes;
}

function parseLineMeta(raw) {
  if (raw == null) return null;
  if (typeof raw === 'string') {
    try {
      const parsed = JSON.parse(raw);
      return parsed && typeof parsed === 'object' ? parsed : null;
    } catch {
      return null;
    }
  }
  return typeof raw === 'object' ? raw : null;
}

export function partnerLineForCustomerNote(line, allLines) {
  const meta = parseLineMeta(line?.line_meta);
  const partnerRid = meta?.round_trip_merge_partner_reservation_id;
  if (partnerRid == null) return null;
  const partnerId = Number(partnerRid);
  if (!Number.isFinite(partnerId)) return null;
  return (Array.isArray(allLines) ? allLines : []).find(
    (other) => Number(other?.reservation_id) === partnerId,
  ) ?? null;
}

export function collectCustomerVisibleNotesForPreview(line, allLines) {
  const sources = [line];
  const partner = partnerLineForCustomerNote(line, allLines);
  if (partner != null && partner !== line) sources.push(partner);
  return collectCustomerVisibleNotes(sources);
}
