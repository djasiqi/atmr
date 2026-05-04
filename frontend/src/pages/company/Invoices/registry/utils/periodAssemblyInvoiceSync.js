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
      await invoiceService.updateDraftInvoiceLine(companyId, invoiceId, keep.id, {
        ...concurrency(inv),
        description: m.description,
        line_total: Number(m.line_total),
        adjustment_note:
          m.adjustment_note != null && String(m.adjustment_note).trim()
            ? String(m.adjustment_note).trim()
            : null,
      });
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
    if (findServerCustomMatch(inv.lines, desc, ht)) continue;

    inv = await loadInvoice(companyId, invoiceId);
    await invoiceService.addDraftCustomLine(companyId, invoiceId, {
      ...concurrency(inv),
      description: desc,
      line_total: ht,
      qty: 1,
    });
  }

  return loadInvoice(companyId, invoiceId);
}
