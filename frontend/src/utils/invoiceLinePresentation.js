/**
 * Présentation unique d'une prestation (patient, institution, partenaire).
 *
 * Mêmes blocs Client et Trajet. « Prise en charge » n'apparaît que si l'option est active.
 */

export const INVOICE_GRAND_TOTAL_LABEL = 'TOTAL À FACTURER :';

const ARTIFICIAL_MIDNIGHT = '00:00';

export function invoiceGrandTotalLabel(_strategy) {
  return INVOICE_GRAND_TOTAL_LABEL;
}

export function usesSharedPrestationLayout(_strategy) {
  return true;
}

function cleanClock(value) {
  const text = String(value || '').trim();
  if (!text || text === ARTIFICIAL_MIDNIGHT) return '';
  return text;
}

/**
 * Libellé d'heure. `actual_boarded` prime. Sinon l'heure prévue, marquée « (prévue) ».
 * `00:00` est absent. Sans heure exploitable : « — ».
 */
export function pickupDisplayLabel({ source, scheduled, boarded, fallback } = {}) {
  const direct = String(fallback || '').trim();
  if (direct) return direct;
  const boardedClock = cleanClock(boarded);
  const scheduledClock = cleanClock(scheduled);
  if (source === 'actual_boarded' && boardedClock) return boardedClock;
  if (scheduledClock && (source === 'scheduled_fallback' || source === 'actual_boarded')) {
    return `${scheduledClock} (prévue)`;
  }
  return '—';
}

function lineMeta(line) {
  const raw = line?.line_meta;
  if (!raw || typeof raw !== 'object') return {};
  return raw;
}

function explicitClientName(line) {
  const meta = lineMeta(line);
  const raw = meta.patient_name || line?.client_name || line?.patient_name || '';
  const name = String(raw).trim();
  if (!name || name === 'Client') return '';
  return name;
}

function stripPartnerTransferPrefix(text) {
  return String(text || '')
    .trim()
    .replace(/^Transfert partenaire\s*[—–-]\s*/i, '')
    .trim();
}

/**
 * @returns {{ clientLine: string|null, pickupLine: string|null, trajetLine: string }}
 */
export function presentPrestationParts(line, { showPickup = false, formatClientName } = {}) {
  const kind = String(line?.type ?? line?.line_type ?? '')
    .trim()
    .toUpperCase();
  const rawDescription = stripPartnerTransferPrefix(line?.description);
  if (kind === 'CUSTOM' || kind === 'MATERIAL_DELIVERY') {
    return {
      clientLine: null,
      pickupLine: null,
      trajetLine: rawDescription || '—',
    };
  }

  const departure = String(line?.departure || lineMeta(line).departure || '').trim();
  const arrival = String(line?.arrival || lineMeta(line).arrival || '').trim();
  let trajetBody = rawDescription;
  let named = explicitClientName(line);
  const split = trajetBody.match(/^(.+?)\s+[—–-]\s+(.+→.+)$/);
  if (split && !/^trajet\s*:/i.test(split[1])) {
    if (!named) named = split[1].trim();
    trajetBody = split[2].trim();
  }
  if (named === 'Client') named = '';
  const formatted = named
    ? String(formatClientName ? formatClientName(named) : named).trim()
    : '';
  const clientLine = formatted ? `Client : ${formatted}` : null;
  let trajetLine;
  if (/^Trajet\s*:/i.test(trajetBody)) {
    trajetLine = trajetBody.replace(/^Trajet\s*:\s*/i, 'Trajet : ');
  } else if (departure || arrival) {
    trajetLine = `Trajet : ${[departure, arrival].filter(Boolean).join(' → ')}`;
  } else if (trajetBody.includes('→')) {
    trajetLine = `Trajet : ${trajetBody}`;
  } else {
    trajetLine = trajetBody || '—';
  }

  let pickupLine = null;
  if (showPickup) {
    const meta = lineMeta(line);
    const label = pickupDisplayLabel({
      source: line?.pickup_time_source || meta.pickup_time_source,
      scheduled: line?.scheduled_pickup_at || meta.scheduled_pickup_at,
      boarded: line?.boarded_at || meta.boarded_at,
      fallback: line?.pickup_label || meta.pickup_label,
    });
    pickupLine = `Prise en charge : ${label}`;
  }

  return { clientLine, pickupLine, trajetLine };
}

function parseMetaObject(raw) {
  if (raw == null) return {};
  if (typeof raw === 'string') {
    try {
      const parsed = JSON.parse(raw);
      return parsed && typeof parsed === 'object' ? parsed : {};
    } catch {
      return {};
    }
  }
  return typeof raw === 'object' ? raw : {};
}

function parseAmount(raw) {
  if (raw == null || raw === '') return null;
  const value = parseFloat(String(raw).replace(',', '.'));
  return Number.isFinite(value) ? value : null;
}

function roundMoney(value) {
  return Math.round((value + Number.EPSILON) * 100) / 100;
}

function formatPercentLabel(value) {
  const rounded = roundMoney(value);
  if (!Number.isFinite(rounded) || rounded <= 0) return '';
  return rounded.toFixed(2).replace(/\.?0+$/, '');
}

function percentExplainsAmounts(catalog, net, percent) {
  if (!Number.isFinite(percent) || percent <= 0 || percent > 100 || catalog <= 0) return false;
  const expected = roundMoney(catalog * (1 - percent / 100));
  return Math.abs(expected - roundMoney(net)) <= 0.05;
}

function storedDiscountPercents(line, invoice) {
  const lineMeta = parseMetaObject(line?.line_meta);
  const invoiceMeta = parseMetaObject(invoice?.meta);
  const found = [];
  const direct = parseAmount(lineMeta.per_line_discount_percent);
  if (direct != null) found.push(direct);
  const rows = invoiceMeta.per_line_discounts?.lines;
  if (Array.isArray(rows) && line?.id != null) {
    const match = rows.find((row) => String(row?.line_id) === String(line.id));
    const fromRows = parseAmount(match?.percent);
    if (fromRows != null) found.push(fromRows);
  }
  const globalPercent = parseAmount(invoiceMeta.global_discount?.percent);
  if (globalPercent != null && lineMeta.original_line_total != null && lineMeta.original_line_total !== '') {
    found.push(globalPercent);
  }
  return found;
}

function discountedAmounts(line) {
  const meta = parseMetaObject(line?.line_meta);
  const catalog = parseAmount(meta.original_line_total);
  const net = parseAmount(line?.line_total);
  if (catalog == null || net == null || Math.abs(catalog - net) < 0.005 || net >= catalog) return null;
  return { catalog, net };
}

function invoiceHasGlobalDiscount(invoice) {
  const discount = parseMetaObject(invoice?.meta).global_discount;
  if (!discount || typeof discount !== 'object') return false;
  const percent = parseAmount(discount.percent);
  const amount = parseAmount(discount.amount_ht);
  return (percent != null && percent > 0) || (amount != null && amount > 0.005);
}

/**
 * Sous-texte d'une remise propre à la ligne.
 * Une réduction globale n'apparaît que dans les totaux, pas sous chaque prestation.
 * @param {object|object[]} lineOrLines
 */
export function lineDiscountCaption(lineOrLines, invoice) {
  if (invoiceHasGlobalDiscount(invoice)) return null;
  const lines = (Array.isArray(lineOrLines) ? lineOrLines : [lineOrLines]).filter(Boolean);
  const parts = lines.map(discountedAmounts).filter(Boolean);
  if (parts.length === 0) return null;
  const catalog = roundMoney(parts.reduce((sum, part) => sum + part.catalog, 0));
  const net = roundMoney(parts.reduce((sum, part) => sum + part.net, 0));
  const deduction = roundMoney(catalog - net);
  if (deduction <= 0) return null;
  const stored = lines
    .flatMap((line) => storedDiscountPercents(line, invoice))
    .find((percent) => percentExplainsAmounts(catalog, net, percent));
  const percent = stored ?? (catalog > 0 ? (deduction / catalog) * 100 : null);
  const percentLabel = formatPercentLabel(percent);
  if (!percentLabel) return null;
  return `Rabais ${percentLabel} % · prix initial ${catalog.toFixed(2)} CHF · -${deduction.toFixed(2)} CHF`;
}

/**
 * Montant de colonne. En réduction globale : prix initial. La remise est dans les totaux.
 * @param {object|null|undefined} partnerLine
 */
export function linePreviewAmount(line, invoice, partnerLine) {
  const netOf = (row) => {
    const value = Number(row?.line_total);
    return Number.isFinite(value) ? value : 0;
  };
  const rows = [line, partnerLine].filter(Boolean);
  const net = roundMoney(rows.reduce((sum, row) => sum + netOf(row), 0));
  if (!invoiceHasGlobalDiscount(invoice)) return net;
  let catalog = 0;
  let sawCatalog = false;
  rows.forEach((row) => {
    const stored = parseAmount(parseMetaObject(row?.line_meta).original_line_total);
    if (stored != null) {
      catalog += stored;
      sawCatalog = true;
    } else {
      catalog += netOf(row);
    }
  });
  return sawCatalog ? roundMoney(catalog) : net;
}
