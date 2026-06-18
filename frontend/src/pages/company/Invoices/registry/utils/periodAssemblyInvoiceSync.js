/**
 * Après génération facture période, aligne le brouillon serveur sur l’aperçu d’assemblage
 * (montants / libellés modifiés, lignes perso, déductions, consolidation A/R une ligne ↔ plusieurs lignes serveur).
 */

import { invoiceService } from '../../../../../services/invoiceService';

function unwrapInvoicePayload(payload) {
  if (!payload || typeof payload !== 'object') return null;
  if (payload.invoice && typeof payload.invoice === 'object' && payload.invoice.id != null) {
    return payload.invoice;
  }
  const inner = payload.data;
  if (inner && typeof inner === 'object') {
    if (inner.invoice && typeof inner.invoice === 'object' && inner.invoice.id != null) {
      return inner.invoice;
    }
    if (inner.id != null && Array.isArray(inner.lines)) {
      return inner;
    }
  }
  if (payload.id != null && Array.isArray(payload.lines)) {
    return payload;
  }
  return null;
}

async function loadInvoice(companyId, invoiceId) {
  const res = await invoiceService.getInvoice(companyId, invoiceId, { cacheBust: true });
  const inv = unwrapInvoicePayload(res) ?? res?.data ?? res;
  if (!inv?.id || !Array.isArray(inv.lines)) {
    throw new Error('INVALID_INVOICE_PAYLOAD');
  }
  return inv;
}

export function bookingIdFromMergedRideLineId(lineId) {
  if (lineId == null) return null;
  const s = String(lineId);
  if (!s.startsWith('pv-')) return null;
  const n = parseInt(s.slice(3), 10);
  return Number.isFinite(n) ? n : null;
}

function isRideLike(typ) {
  const t = String(typ || '').toLowerCase();
  return t === 'ride' || t === 'material_delivery';
}

function concurrency(inv) {
  return inv?.updated_at ? { expected_updated_at: inv.updated_at } : {};
}

function findServerCustomMatch(serverLines, description, lineTotal) {
  const desc = String(description || '').trim();
  const ht = Number(lineTotal);
  return serverLines.find((l) => {
    if (String(l.type || '').toLowerCase() !== 'custom') return false;
    if (l.reservation_id != null) return false;
    return (
      String(l.description || '').trim() === desc && Math.abs(Number(l.line_total) - ht) < 0.005
    );
  });
}

/** Mêmes clés que le backend (PATCH `line_meta_merge`) : A/R, fusion A/R, etc. */
function previewTransportMetaPick(lineMeta) {
  const m = lineMeta && typeof lineMeta === 'object' ? lineMeta : {};
  const o = {};
  if (m.is_round_trip_leg === true) o.is_round_trip_leg = true;
  const tt = String(m.transport_type || '').trim();
  if (tt) o.transport_type = tt;
  if (m.preview_hide_merged_round_trip === true) o.preview_hide_merged_round_trip = true;
  if (
    m.round_trip_merge_partner_reservation_id != null &&
    Number.isFinite(Number(m.round_trip_merge_partner_reservation_id))
  ) {
    o.round_trip_merge_partner_reservation_id = Number(m.round_trip_merge_partner_reservation_id);
  }
  return o;
}

function transportMetaPatchIfNeeded(mergedLineMeta, serverLine) {
  const want = previewTransportMetaPick(mergedLineMeta);
  if (!Object.keys(want).length) return null;
  const slm =
    serverLine?.line_meta && typeof serverLine.line_meta === 'object' ? serverLine.line_meta : {};
  const have = previewTransportMetaPick(slm);
  const patch = {};
  for (const k of Object.keys(want)) {
    if (have[k] !== want[k]) patch[k] = want[k];
  }
  return Object.keys(patch).length ? patch : null;
}

/** Jambe A/R déjà exclue dans l’aperçu période : retirer les marqueurs fusion sur le brouillon. */
function roundTripLegExcludedMetaPatch(previewMeta) {
  const m = previewMeta && typeof previewMeta === 'object' ? previewMeta : {};
  if (!m.period_preview_single_leg) return null;
  return {
    is_round_trip_leg: null,
    transport_type: null,
    preview_hide_merged_round_trip: null,
    round_trip_merge_partner_reservation_id: null,
  };
}

function transportMetaSyncPatch(mergedLineMeta, serverLine) {
  const legPatch = roundTripLegExcludedMetaPatch(mergedLineMeta);
  if (legPatch) return legPatch;
  return transportMetaPatchIfNeeded(mergedLineMeta, serverLine);
}

function serviceDateIsoFromLineMeta(lineMeta) {
  const m = lineMeta && typeof lineMeta === 'object' ? lineMeta : {};
  const raw = [m.service_date_iso, m.service_date].find(
    (x) => x != null && String(x).trim() !== ''
  );
  return raw != null ? String(raw).trim().slice(0, 10) : '';
}

/**
 * @param {number} companyId
 * @param {number} invoiceId
 * @param {object} mergedInvoice — sortie de mergePeriodPreviewInvoice (aperçu HTML)
 * @returns {Promise<object>} facture JSON à jour
 */
export async function syncDraftInvoiceWithMergedAssemblyPreview(companyId, invoiceId, mergedInvoice) {
  let inv = await loadInvoice(companyId, invoiceId);
  const mergedLines = Array.isArray(mergedInvoice?.lines) ? mergedInvoice.lines : [];

  const mergedRides = mergedLines.filter((l) => isRideLike(l.type || l.line_type));
  let serverRides = inv.lines.filter((l) => isRideLike(l.type) && l.reservation_id != null);

  // Consolidation A/R : une ligne dans l’aperçu, plusieurs lignes réservation côté serveur (même total HT)
  if (mergedRides.length === 1 && serverRides.length > 1) {
    const m = mergedRides[0];
    const sumHt = serverRides.reduce((s, l) => s + Number(l.line_total || 0), 0);
    if (Math.abs(sumHt - Number(m.line_total || 0)) < 0.02) {
      const sorted = [...serverRides].sort((a, b) => Number(a.id) - Number(b.id));
      const [keep, ...drop] = sorted;
      inv = await loadInvoice(companyId, invoiceId);
      const km = m.line_meta && typeof m.line_meta === 'object' ? m.line_meta : {};
      const mergeBody = {
        ...concurrency(inv),
        description: m.description,
        line_total: Number(m.line_total),
        adjustment_note:
          m.adjustment_note != null && String(m.adjustment_note).trim()
            ? String(m.adjustment_note).trim()
            : null,
      };
      if (km.original_line_total != null && Number.isFinite(Number(km.original_line_total))) {
        mergeBody.original_line_total = Number(km.original_line_total);
      }
      const tp = transportMetaSyncPatch(km, keep);
      if (tp) {
        mergeBody.line_meta_merge = tp;
      }
      const wantSd = serviceDateIsoFromLineMeta(km);
      if (wantSd) {
        mergeBody.service_date_iso = wantSd;
      }
      await invoiceService.updateDraftInvoiceLine(companyId, invoiceId, keep.id, mergeBody);
      inv = await loadInvoice(companyId, invoiceId);
      for (const row of drop) {
        inv = await loadInvoice(companyId, invoiceId);
        await invoiceService.removeDraftInvoiceLine(companyId, invoiceId, row.id, {
          expected_updated_at: inv.updated_at,
        });
      }
      inv = await loadInvoice(companyId, invoiceId);
    }
  }

  // PATCH trajets / livraisons : réservation ↔ pv-{booking_id}
  for (const ml of mergedLines) {
    if (!isRideLike(ml.type || ml.line_type)) continue;
    const bid = bookingIdFromMergedRideLineId(ml.id);
    if (bid == null) continue;
    inv = await loadInvoice(companyId, invoiceId);
    const sl = inv.lines.find(
      (l) =>
        isRideLike(l.type) &&
        l.reservation_id != null &&
        String(l.reservation_id) === String(bid)
    );
    if (!sl) continue;

    const body = { ...concurrency(inv) };
    let changed = false;
    if (String(ml.description || '') !== String(sl.description || '')) {
      body.description = ml.description;
      changed = true;
    }
    if (Math.abs(Number(ml.line_total) - Number(sl.line_total)) > 0.004) {
      body.line_total = Number(ml.line_total);
      changed = true;
    }
    const mn = ml.adjustment_note ?? '';
    const sn = sl.adjustment_note ?? '';
    if (String(mn).trim() !== String(sn).trim()) {
      body.adjustment_note = mn.trim() ? mn : null;
      changed = true;
    }
    const mlm = ml.line_meta && typeof ml.line_meta === 'object' ? ml.line_meta : {};
    const slm = sl.line_meta && typeof sl.line_meta === 'object' ? sl.line_meta : {};
    const mOlt = mlm.original_line_total;
    const sOlt = slm.original_line_total;
    if (
      mOlt != null &&
      Number.isFinite(Number(mOlt)) &&
      (sOlt == null || Math.abs(Number(sOlt) - Number(mOlt)) > 0.004)
    ) {
      body.original_line_total = Number(mOlt);
      changed = true;
    }
    const tPatch = transportMetaSyncPatch(mlm, sl);
    if (tPatch) {
      body.line_meta_merge = tPatch;
      changed = true;
    }
    const wantSd = serviceDateIsoFromLineMeta(mlm);
    const haveSd = serviceDateIsoFromLineMeta(slm);
    if (wantSd && wantSd !== haveSd) {
      body.service_date_iso = wantSd;
      changed = true;
    }
    if (changed) {
      await invoiceService.updateDraftInvoiceLine(companyId, invoiceId, sl.id, body);
    }
  }

  // Lignes CUSTOM (prestations + déductions HT négatives)
  inv = await loadInvoice(companyId, invoiceId);
  for (const ml of mergedLines) {
    if (String(ml.type || ml.line_type || '').toLowerCase() !== 'custom') continue;
    const ht = Number(ml.line_total);
    if (!Number.isFinite(ht) || ht === 0) continue;
    const desc = String(ml.description || '—').trim().slice(0, 500);
    inv = await loadInvoice(companyId, invoiceId);
    const existing = findServerCustomMatch(inv.lines, desc, ht);
    if (existing) {
      const lm = ml.line_meta && typeof ml.line_meta === 'object' ? ml.line_meta : {};
      const want =
        [lm.service_date_iso, lm.service_date].find((x) => x != null && String(x).trim()) || '';
      const slm =
        existing.line_meta && typeof existing.line_meta === 'object' ? existing.line_meta : {};
      const have =
        [slm.service_date_iso, slm.service_date].find((x) => x != null && String(x).trim()) || '';
      if (String(want).trim() && String(want).trim() !== String(have).trim()) {
        await invoiceService.updateDraftInvoiceLine(companyId, invoiceId, existing.id, {
          ...concurrency(inv),
          service_date_iso: String(want).trim().slice(0, 10),
        });
      }
      continue;
    }

    inv = await loadInvoice(companyId, invoiceId);
    const lm = ml.line_meta && typeof ml.line_meta === 'object' ? ml.line_meta : {};
    const cp = lm.custom_prestation;
    const modeQty = cp && cp.mode === 'quantity';
    const qtyNum = Number(ml.qty);
    const qtyPayload = Number.isFinite(qtyNum) && qtyNum > 0 ? qtyNum : 1;
    const body = {
      ...concurrency(inv),
      description: desc,
      line_total: ht,
      qty: qtyPayload,
      custom_mode: modeQty ? 'quantity' : 'time',
    };
    if (!modeQty) {
      body.time_unit = (cp && cp.time_unit) || 'h';
    }
    const sdi = lm.service_date_iso || lm.service_date;
    if (sdi != null && String(sdi).trim()) {
      body.service_date_iso = String(sdi).trim();
    }
    await invoiceService.addDraftCustomLine(companyId, invoiceId, body);
  }

  return loadInvoice(companyId, invoiceId);
}
