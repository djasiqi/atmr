/**
 * Préparation corrective locale (front) — aucune mutation de l'original.
 * Les montants officiels d'avoir restent validés par l'API preview/issue.
 */

export function money(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return 0;
  return Math.round(n * 100) / 100;
}

/** Clone superficiel facture + lignes pour édition locale. */
export function cloneInvoiceForCorrection(invoice) {
  if (!invoice || typeof invoice !== 'object') return null;
  const lines = Array.isArray(invoice.lines)
    ? invoice.lines.map((line) => ({
        ...line,
        line_meta:
          line.line_meta && typeof line.line_meta === 'object'
            ? { ...line.line_meta }
            : line.line_meta,
      }))
    : [];
  return {
    ...invoice,
    lines,
    _correction_prep: true,
  };
}

function lineTtc(line) {
  const ttc = Number(line?.total_with_vat);
  if (Number.isFinite(ttc)) return money(ttc);
  const ht = Number(line?.line_total);
  const vat = Number(line?.vat_amount);
  if (Number.isFinite(ht) && Number.isFinite(vat)) return money(ht + vat);
  return money(ht);
}

function estimateVatFromHt(ht, vatRate) {
  const rate = Number(vatRate);
  if (!Number.isFinite(ht) || !Number.isFinite(rate)) return 0;
  return money((ht * rate) / 100);
}

/**
 * Applique un PATCH local (libellé / HT / note) sur la préparation.
 * Recalcule TVA/TTC approximatifs pour l'aperçu ; le serveur tranche à la confirmation.
 */
export function applyLocalLinePatch(invoice, lineId, patch) {
  const next = cloneInvoiceForCorrection(invoice);
  if (!next) return invoice;
  next.lines = next.lines.map((line) => {
    if (Number(line.id) !== Number(lineId)) return line;
    const updated = { ...line };
    if (patch.description !== undefined) {
      updated.description = String(patch.description || '').slice(0, 500);
    }
    if (patch.adjustment_note !== undefined) {
      updated.adjustment_note = patch.adjustment_note;
    }
    if (patch.line_total !== undefined) {
      const ht = money(patch.line_total);
      const rate = Number(line.vat_rate ?? 7.7);
      const vat = estimateVatFromHt(ht, rate);
      updated.line_total = ht;
      updated.unit_price = ht;
      updated.vat_amount = vat;
      updated.total_with_vat = money(ht + vat);
    }
    return updated;
  });
  return recomputePrepTotals(next);
}

export function removeLocalLine(invoice, lineId) {
  const next = cloneInvoiceForCorrection(invoice);
  if (!next) return invoice;
  next.lines = next.lines.filter((line) => Number(line.id) !== Number(lineId));
  return recomputePrepTotals(next);
}

function isRemisablePrepLine(line) {
  const ht = Number(line?.line_total);
  if (!Number.isFinite(ht) || ht <= 0) return false;
  const m = line?.line_meta || {};
  if (m.manual_discount || m.global_discount_line || m.per_line_discount_line) {
    return false;
  }
  const t = String(line?.type || '').toLowerCase();
  // ride / custom / type absent (préparation locale) : remisable
  return t === 'ride' || t === 'custom' || t === '';
}

/**
 * Remise % locale sur la préparation (n’écrit pas l’original).
 * Réduit le HT de chaque ligne remisable ; le crédit exact est tranché serveur.
 */
export function applyLocalPercentDiscount(invoice, percent) {
  const p = Number(percent);
  if (!Number.isFinite(p) || p <= 0 || p > 100) {
    return { invoice, error: 'Pourcentage de remise invalide (0–100).' };
  }
  let next = cloneInvoiceForCorrection(invoice);
  if (!next) return { invoice, error: 'Préparation indisponible.' };
  let touched = 0;
  next.lines = next.lines.map((line) => {
    if (!isRemisablePrepLine(line)) return line;
    const ht = money(line.line_total);
    const reduced = money(ht * (1 - p / 100));
    touched += 1;
    const rate = Number(line.vat_rate ?? 7.7);
    const vat = estimateVatFromHt(reduced, rate);
    return {
      ...line,
      line_total: reduced,
      unit_price: reduced,
      vat_amount: vat,
      total_with_vat: money(reduced + vat),
      line_meta: {
        ...(line.line_meta && typeof line.line_meta === 'object' ? line.line_meta : {}),
        correction_percent_discount: p,
      },
    };
  });
  if (touched === 0) {
    return { invoice, error: 'Aucune ligne HT remisable dans la préparation.' };
  }
  next = recomputePrepTotals(next);
  return { invoice: next, error: null, touched };
}

export function addLocalCustomLine(invoice, { description, line_total, service_date }) {
  const next = cloneInvoiceForCorrection(invoice);
  if (!next) return invoice;
  const ht = money(line_total);
  const vat = estimateVatFromHt(ht, 7.7);
  const tempId = -Math.abs(Date.now());
  next.lines = [
    ...next.lines,
    {
      id: tempId,
      type: 'custom',
      description: String(description || '').slice(0, 500),
      qty: 1,
      unit_price: ht,
      line_total: ht,
      vat_rate: 7.7,
      vat_amount: vat,
      total_with_vat: money(ht + vat),
      reservation_id: null,
      service_date: service_date || null,
      line_meta: { correction_added: true, service_date: service_date || null },
      adjustment_note: null,
    },
  ];
  return recomputePrepTotals(next);
}

export function recomputePrepTotals(invoice) {
  const next = cloneInvoiceForCorrection(invoice);
  if (!next) return invoice;
  let ht = 0;
  let vat = 0;
  let ttc = 0;
  next.lines.forEach((line) => {
    ht += money(line.line_total);
    vat += money(line.vat_amount);
    ttc += lineTtc(line);
  });
  next.subtotal_amount = money(ht);
  next.vat_total_amount = money(vat);
  next.total_amount = money(ttc);
  next.balance_due = money(ttc - money(next.amount_paid));
  return next;
}

/**
 * Construit le payload `lines` pour preview/issue credit-note à partir
 * de l'original + résiduels serveur + préparation locale.
 *
 * Règle sûre (API existante) :
 * - ligne absente de la prep → crédit = résiduel TTC
 * - ligne avec TTC cible < résiduel restant sur montant original créditable → crédit partiel
 * - nouvelles lignes (id < 0) → ignorées pour l'avoir (rectificative seulement après)
 * - libellé seul sans baisse de montant → erreur explicite (non représentable sans crédit)
 */
export function buildCreditLinesFromPreparation({
  originalInvoice,
  preparationInvoice,
  residuals,
}) {
  const residualById = new Map(
    (residuals || []).map((row) => [Number(row.invoice_line_id), row])
  );
  const prepById = new Map(
    (preparationInvoice?.lines || [])
      .filter((l) => Number(l.id) > 0)
      .map((l) => [Number(l.id), l])
  );
  const creditLines = [];
  const warnings = [];
  const unsupported = [];

  residualById.forEach((row, lineId) => {
    const residual = money(row.residual_ttc);
    if (residual <= 0) return;
    const prep = prepById.get(lineId);
    if (!prep) {
      creditLines.push({ invoice_line_id: lineId });
      return;
    }
    const targetTtc = lineTtc(prep);
    const originalLineTtc = money(row.amount_ttc);
    const already = money(row.credited_ttc);
    const remainingOriginal = money(originalLineTtc - already);
    // Cible par rapport au résiduel encore créditable sur cette ligne
    const maxKeep = residual;
    if (targetTtc >= maxKeep - 0.001) {
      const origDesc = String(row.description || '');
      const nextDesc = String(prep.description || '');
      if (nextDesc && nextDesc !== origDesc) {
        unsupported.push({
          invoice_line_id: lineId,
          reason:
            'Changement de libellé seul sans réduction de montant : non représentable par un avoir partiel. Réduisez le montant ou retirez la ligne.',
        });
      }
      return;
    }
    if (targetTtc <= 0) {
      creditLines.push({ invoice_line_id: lineId });
      return;
    }
    const creditTtc = money(residual - targetTtc);
    if (creditTtc <= 0) return;
    if (creditTtc >= residual - 0.001) {
      creditLines.push({ invoice_line_id: lineId });
    } else {
      creditLines.push({
        invoice_line_id: lineId,
        amount_ttc: String(creditTtc),
      });
    }
    if (remainingOriginal < residual) {
      warnings.push(`Ligne ${lineId}: résiduel plafonné par le serveur.`);
    }
  });

  const added = (preparationInvoice?.lines || []).filter((l) => Number(l.id) < 0);
  if (added.length > 0) {
    warnings.push(
      `${added.length} ligne(s) ajoutée(s) : elles ne figureront sur la rectificative qu'après émission de l'avoir (édition du brouillon rectificatif).`
    );
  }

  return { creditLines, warnings, unsupported };
}

export function newIdempotencyKey(prefix, invoiceId) {
  const rand =
    typeof crypto !== 'undefined' && crypto.randomUUID
      ? crypto.randomUUID().slice(0, 8)
      : String(Date.now());
  return `${prefix}-${invoiceId}-${Date.now()}-${rand}`.slice(0, 80);
}
