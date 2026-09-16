/**
 * Identité typée des factures du registre company.
 *
 * Les séquences `invoices.id` et `partner_invoices.id` sont indépendantes :
 * un même entier (ex. 34) n’identifie pas la même ressource. Le type doit
 * voyager avec l’id (objet, query `invoice_type`, resolver unique).
 */

import {
  isPartnerInvoice,
  resolveInvoicePdfApiUrl,
} from './pdfUrlFallback';

export const INVOICE_CATALOG = {
  STANDARD: 'standard',
  PARTNER: 'partner',
};

/**
 * @param {object|null|undefined} invoice
 * @param {{ type?: string, invoice_type?: string }|null|undefined} [extras]
 * @returns {'standard'|'partner'}
 */
export function resolveInvoiceCatalogType(invoice, extras = {}) {
  const hinted = String(extras?.type || extras?.invoice_type || '').toLowerCase();
  if (hinted === INVOICE_CATALOG.PARTNER) return INVOICE_CATALOG.PARTNER;
  if (hinted === INVOICE_CATALOG.STANDARD) {
    return isPartnerInvoice(invoice) ? INVOICE_CATALOG.PARTNER : INVOICE_CATALOG.STANDARD;
  }
  return isPartnerInvoice(invoice) ? INVOICE_CATALOG.PARTNER : INVOICE_CATALOG.STANDARD;
}

/**
 * @param {object|null|undefined} invoice
 * @param {'standard'|'partner'|null|undefined} [type]
 * @returns {object|null|undefined}
 */
export function stampInvoiceCatalog(invoice, type) {
  if (!invoice || typeof invoice !== 'object') return invoice;
  const resolved = type || resolveInvoiceCatalogType(invoice);
  if (resolved === INVOICE_CATALOG.PARTNER) {
    return {
      ...invoice,
      is_partner_invoice: true,
      kind: 'partner',
      invoice_type: INVOICE_CATALOG.PARTNER,
    };
  }
  return {
    ...invoice,
    is_partner_invoice: false,
    kind: invoice.kind && invoice.kind !== 'partner' ? invoice.kind : 'standard',
    invoice_type: INVOICE_CATALOG.STANDARD,
  };
}

/**
 * Empreinte d’identité (id + catalogue) pour invalider un GET standard périmé.
 * @param {object|null|undefined} invoice
 * @returns {string}
 */
export function invoiceCatalogFingerprint(invoice) {
  if (!invoice || invoice.id == null) return '';
  return `${resolveInvoiceCatalogType(invoice)}:${invoice.id}:${String(invoice.invoice_number || '')}`;
}

/**
 * Clé de ligne / sélection. Ne jamais utiliser `invoice.id` seul
 * (collision entre catalogues).
 * @param {object|null|undefined} invoice
 * @param {number|string|null|undefined} [companyId]
 * @returns {string}
 */
export function invoiceCatalogRowKey(invoice, companyId) {
  const resource = resolveInvoiceResource(invoice, companyId);
  if (resource.id == null) return '';
  return `${resource.type}:${resource.id}`;
}

/**
 * @param {URLSearchParams|null|undefined} searchParams
 * @returns {{ type: 'standard'|'partner', id: number }|null}
 */
export function readInvoiceCatalogFromSearch(searchParams) {
  if (!searchParams || typeof searchParams.get !== 'function') return null;
  const idRaw = searchParams.get('invoice_id');
  if (idRaw == null || String(idRaw).trim() === '') return null;
  const id = Number(idRaw);
  if (!Number.isFinite(id)) return null;
  const typeRaw = String(searchParams.get('invoice_type') || '').toLowerCase();
  const type =
    typeRaw === INVOICE_CATALOG.PARTNER
      ? INVOICE_CATALOG.PARTNER
      : INVOICE_CATALOG.STANDARD;
  return { type, id };
}

/**
 * @param {URLSearchParams} params
 * @param {{ type: string, id: number|string }} resource
 * @returns {URLSearchParams}
 */
export function writeInvoiceCatalogSearchParams(params, { type, id }) {
  if (!params || id == null) return params;
  params.set('invoice_id', String(id));
  params.set(
    'invoice_type',
    type === INVOICE_CATALOG.PARTNER
      ? INVOICE_CATALOG.PARTNER
      : INVOICE_CATALOG.STANDARD
  );
  return params;
}

/**
 * @param {URLSearchParams} params
 * @returns {URLSearchParams}
 */
export function clearInvoiceCatalogSearchParams(params) {
  if (!params) return params;
  params.delete('invoice_id');
  params.delete('invoice_type');
  params.delete('draft_edit');
  params.delete('partner');
  return params;
}

/**
 * Resolver unique : type + endpoints. Aucun écran ne doit construire
 * `/invoices/{id}` à partir d’un id nu.
 *
 * @param {object|null|undefined} invoice
 * @param {number|string|null|undefined} [companyId]
 * @param {{ type?: string, invoice_type?: string, id?: number }} [extras]
 * @returns {{
 *   type: 'standard'|'partner',
 *   id: number|null,
 *   companyId: number|string|null,
 *   invoice: object|null,
 *   pdfApiUrl: string|null,
 *   allowsStandardDetailGet: boolean,
 * }}
 */
export function resolveInvoiceResource(invoice, companyId, extras = {}) {
  const id = invoice?.id ?? extras?.id ?? null;
  const cid = companyId || invoice?.company_id || null;
  const type = resolveInvoiceCatalogType(
    invoice?.id != null ? invoice : { ...invoice, id },
    extras
  );
  const stamped =
    id == null
      ? null
      : stampInvoiceCatalog({ ...(invoice || {}), id, company_id: cid }, type);
  return {
    type,
    id,
    companyId: cid,
    invoice: stamped,
    pdfApiUrl: stamped ? resolveInvoicePdfApiUrl(stamped, cid) : null,
    allowsStandardDetailGet: type === INVOICE_CATALOG.STANDARD && id != null,
  };
}

/**
 * Session de chargement : une requête standard en vol est ignorée si le
 * catalogue change (partenaire résolu) ou si un nouveau chargement commence.
 * @returns {{ begin: () => number, isCurrent: (token: number) => boolean, invalidate: () => void }}
 */
export function createInvoiceLoadSession() {
  let seq = 0;
  return {
    begin() {
      seq += 1;
      return seq;
    },
    isCurrent(token) {
      return token === seq;
    },
    invalidate() {
      seq += 1;
    },
  };
}

/**
 * Plan d’ouverture d’une facture déjà existante (409 prepare).
 * Un numéro PARTNER- ne déclenche jamais GET /invoices/{id}.
 *
 * @param {{ existingInvoiceId: number, existingInvoiceNumber?: string, companyId?: number }} input
 */
export function existingInvoiceOpenPlan({
  existingInvoiceId,
  existingInvoiceNumber,
  companyId,
}) {
  const resource = resolveInvoiceResource(
    {
      id: existingInvoiceId,
      invoice_number: existingInvoiceNumber,
      company_id: companyId,
    },
    companyId
  );
  return {
    resource,
    fetchStandardDetail: resource.allowsStandardDetailGet,
    searchParams: {
      search: existingInvoiceNumber || '',
      focusSearch: '1',
      invoice_id: String(existingInvoiceId),
      invoice_type: resource.type,
    },
  };
}
