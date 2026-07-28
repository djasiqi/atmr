/**
 * Nom de fichier PDF facture personnalisé (téléchargement / onglet).
 * Ex. : Facture_Juillet_2026_EM-2026-07-0005_M_VUILLE_Michel_155CHF.pdf
 */

const MONTHS_FR = [
  'Janvier',
  'Fevrier',
  'Mars',
  'Avril',
  'Mai',
  'Juin',
  'Juillet',
  'Aout',
  'Septembre',
  'Octobre',
  'Novembre',
  'Decembre',
];

/**
 * @param {unknown} value
 * @param {number} [maxLen]
 * @returns {string}
 */
export function slugifyInvoiceFilenamePart(value, maxLen = 48) {
  if (value == null) return '';
  const ascii = String(value)
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[^a-zA-Z0-9]+/g, '_')
    .replace(/_+/g, '_')
    .replace(/^_|_$/g, '');
  if (!ascii) return '';
  return ascii.slice(0, maxLen);
}

/**
 * @param {unknown} amount
 * @returns {string} ex. 155CHF ou 155_50CHF
 */
export function formatInvoiceAmountForFilename(amount) {
  const n = Number(amount);
  if (!Number.isFinite(n)) return '0CHF';
  const rounded = Math.round(n * 100) / 100;
  if (Math.abs(rounded - Math.round(rounded)) < 1e-9) {
    return `${Math.round(rounded)}CHF`;
  }
  return `${rounded.toFixed(2).replace('.', '_')}CHF`;
}

/**
 * Libellé client / payeur pour le nom de fichier.
 * @param {object|null|undefined} invoice
 * @returns {string}
 */
export function resolveInvoiceFilenameClientLabel(invoice) {
  if (!invoice || typeof invoice !== 'object') return 'Client';

  const billingParty = invoice.billing_party?.display_name;
  if (billingParty && String(billingParty).trim()) {
    return String(billingParty).trim();
  }

  const companyName = invoice.billed_to_company?.name;
  if (companyName && String(companyName).trim()) {
    return String(companyName).trim();
  }

  const billTo = invoice.bill_to_client;
  if (billTo) {
    if (billTo.is_institution && billTo.institution_name) {
      return String(billTo.institution_name).trim();
    }
    const last = String(billTo.last_name || '').trim();
    const first = String(billTo.first_name || '').trim();
    if (last || first) return [last, first].filter(Boolean).join(' ');
    if (billTo.username) return String(billTo.username).trim();
  }

  const client = invoice.client;
  if (client) {
    if (client.patient_display_name) {
      return String(client.patient_display_name).trim();
    }
    if (client.is_institution && client.institution_name) {
      return String(client.institution_name).trim();
    }
    const last = String(client.last_name || '').trim();
    const first = String(client.first_name || '').trim();
    if (last || first) return [last, first].filter(Boolean).join(' ');
    if (client.display_name) return String(client.display_name).trim();
    if (client.username) return String(client.username).trim();
  }

  return 'Client';
}

/**
 * @param {object|null|undefined} invoice
 * @returns {string}
 */
export function buildInvoicePdfDownloadFilename(invoice) {
  const monthIdx = Number(invoice?.period_month);
  const year = Number(invoice?.period_year);
  const monthLabel =
    Number.isInteger(monthIdx) && monthIdx >= 1 && monthIdx <= 12
      ? MONTHS_FR[monthIdx - 1]
      : 'Periode';
  const yearLabel = Number.isFinite(year) && year > 0 ? String(year) : '';

  const numberRaw = String(invoice?.invoice_number || '').trim();
  /** Conserver les tirets du n° (ex. EM-2026-07-0005), retirer seulement les caractères dangereux. */
  const numberPart =
    numberRaw
      .normalize('NFD')
      .replace(/[\u0300-\u036f]/g, '')
      .replace(/[/\\?%*:|"<>]/g, '-')
      .replace(/\s+/g, '_')
      .replace(/_+/g, '_')
      .replace(/^_|_$/g, '')
      .slice(0, 64) ||
    (invoice?.id != null ? `ID_${invoice.id}` : 'SansNumero');

  const clientPart =
    slugifyInvoiceFilenamePart(resolveInvoiceFilenameClientLabel(invoice), 56) ||
    'Client';

  const amountPart = formatInvoiceAmountForFilename(invoice?.total_amount);

  const parts = ['Facture', monthLabel];
  if (yearLabel) parts.push(yearLabel);
  parts.push(numberPart, clientPart, amountPart);

  return `${parts.join('_')}.pdf`;
}
