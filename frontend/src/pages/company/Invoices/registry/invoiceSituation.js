/**
 * Consultation de la situation financière.
 * Affiche les chaînes renvoyées par l'API. Aucun solde n'est recalculé ici.
 */
import { isPartnerInvoice } from '../../../../utils/pdfUrlFallback';

export function formatServerAmount(value) {
  if (value == null || value === '') return 'Montant non indiqué';
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) return 'Montant non indiqué';
    return `${value.toFixed(2)} CHF`;
  }
  const text = String(value).trim();
  if (!text) return 'Montant non indiqué';
  return text.endsWith('CHF') ? text : `${text} CHF`;
}

export function readStatement(payload) {
  if (!payload || typeof payload !== 'object') return null;
  if (payload.statement) return payload.statement;
  if (payload.data && payload.data.statement) return payload.data.statement;
  return null;
}

export async function loadInvoiceSituation({ invoiceService, companyId, invoice }) {
  if (!companyId || invoice?.id == null) {
    return { statement: null, error: 'Facture introuvable.' };
  }
  const response = isPartnerInvoice(invoice)
    ? await invoiceService.getPartnerInvoice(companyId, invoice.id)
    : await invoiceService.getInvoice(companyId, invoice.id);
  const statement = readStatement(response);
  if (!statement || statement.disponible === false) {
    return {
      statement: null,
      error: statement?.message || 'Situation indisponible.',
    };
  }
  return { statement, error: null };
}
