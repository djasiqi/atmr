/**
 * Capacités métier par catalogue. Le menu ne masque plus une action
 * simplement parce que `is_partner_invoice === true`.
 */
import { INVOICE_CATALOG, resolveInvoiceResource } from './invoiceCatalog';
import {
  canAddPayment,
  canCancelInvoice,
  canDuplicateInvoice,
  canEditDraft,
  canGenerateReminder,
  canRegeneratePdf,
  canSendInvoice,
  invoiceStatusLower,
} from '../services/invoiceService';

/**
 * @param {object|null|undefined} invoice
 * @param {number|string|null|undefined} [companyId]
 */
export function getInvoiceCapabilities(invoice, companyId) {
  const resource = resolveInvoiceResource(invoice, companyId);
  const status = invoiceStatusLower(invoice);
  const paid = status === 'paid';
  const cancelled = status === 'cancelled';
  const draft = status === 'draft';
  const amountPaid = Number(invoice?.amount_paid || 0);

  if (resource.type === INVOICE_CATALOG.PARTNER) {
    const editable = ['draft', 'sent', 'partially_paid', 'overdue'].includes(status);
    return {
      resource,
      canEdit: editable,
      canViewPdf: true,
      canSend: draft,
      canAddPayment: !paid && !cancelled,
      canGenerateReminder: false,
      canSendReminderEmail: false,
      canViewReminder: false,
      canRegeneratePdf: !paid && !cancelled,
      canCancel: (draft || status === 'sent' || status === 'overdue') && amountPaid === 0,
      canDuplicate: false,
    };
  }

  return {
    resource,
    canEdit: canEditDraft(invoice),
    canViewPdf: true,
    canSend: canSendInvoice(invoice),
    canAddPayment: canAddPayment(invoice),
    canGenerateReminder: canGenerateReminder(invoice),
    canSendReminderEmail:
      Number(invoice?.reminder_level || 0) > 0 && !paid && !cancelled,
    canViewReminder: Boolean(invoice?.reminders?.length),
    canRegeneratePdf: canRegeneratePdf(invoice),
    canCancel: canCancelInvoice(invoice),
    canDuplicate: canDuplicateInvoice(invoice),
  };
}
