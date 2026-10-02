/**
 * Adaptateur d'édition : un contrat UI, deux catalogues backend.
 * Le panneau ne construit jamais /invoices/{id} ni /partner-invoices/{id}.
 */
import {
  INVOICE_CATALOG,
  resolveInvoiceResource,
} from '../../../../utils/invoiceCatalog';
import {
  getInvoice,
  invoiceService,
} from '../../../../services/invoiceService';

function asInvoiceRecord(payload) {
  if (!payload || typeof payload !== 'object') return null;
  if (payload.invoice && payload.invoice.id != null) return payload.invoice;
  const inner = payload.data;
  if (inner && typeof inner === 'object') {
    if (inner.invoice && inner.invoice.id != null) return inner.invoice;
    if (inner.id != null) return inner;
  }
  if (payload.id != null) return payload;
  return null;
}

function mapPartnerLine(line) {
  if (!line || typeof line !== 'object') return line;
  const source = String(line.source_type || '');
  const type =
    line.type ||
    (source === 'custom' || source === 'manual_discount' ? 'custom' : 'ride');
  const amount = line.line_total != null ? line.line_total : line.amount;
  const serviceDate = line.service_date || line.line_meta?.service_date || null;
  return {
    ...line,
    type,
    line_total: amount != null ? Number(amount) : null,
    adjustment_note: line.adjustment_note ?? null,
    line_meta: {
      ...(line.line_meta && typeof line.line_meta === 'object' ? line.line_meta : {}),
      ...(serviceDate ? { service_date: serviceDate } : {}),
    },
  };
}

/** DTO UI commun. Le catalogue reste tamponné pour éviter toute collision d'id. */
export function normalizeInvoiceDraft(raw, catalogType) {
  const record = asInvoiceRecord(raw) || raw;
  if (!record || record.id == null) return record;
  if (catalogType === INVOICE_CATALOG.PARTNER) {
    const lines = Array.isArray(record.lines) ? record.lines.map(mapPartnerLine) : [];
    return {
      ...record,
      catalog_type: INVOICE_CATALOG.PARTNER,
      is_partner_invoice: true,
      kind: 'partner',
      invoice_type: INVOICE_CATALOG.PARTNER,
      billing_strategy: record.billing_strategy || 'partner_monthly',
      editor_header: true,
      subject_contact: record.subject_contact || record.recipient_contact || '',
      vat_total_amount:
        record.vat_total_amount != null ? record.vat_total_amount : record.vat_amount,
      lines,
    };
  }
  return {
    ...record,
    catalog_type: INVOICE_CATALOG.STANDARD,
    is_partner_invoice: false,
    invoice_type: INVOICE_CATALOG.STANDARD,
    editor_header: false,
    lines: Array.isArray(record.lines) ? record.lines : [],
  };
}

function wrapInvoice(invoice) {
  return invoice ? { invoice } : null;
}

/**
 * @param {object|null|undefined} invoice
 * @param {number|string|null|undefined} companyId
 */
export function createInvoiceDraftAdapter(invoice, companyId) {
  const resource = resolveInvoiceResource(invoice, companyId);
  const partner = resource.type === INVOICE_CATALOG.PARTNER;
  const cid = resource.companyId;
  const id = resource.id;

  const normalize = (payload) =>
    normalizeInvoiceDraft(asInvoiceRecord(payload) || payload, resource.type);

  return {
    catalogType: resource.type,
    resource,

    pdfApiUrl(current) {
      if (!resource.pdfApiUrl) return null;
      if (partner) return resource.pdfApiUrl;
      const url = current?.pdf_url ?? resource.invoice?.pdf_url;
      if (!String(url || '').trim()) return null;
      return resource.pdfApiUrl;
    },

    async load() {
      if (!cid || id == null) throw new Error('MISSING_INVOICE_CONTEXT');
      const raw = partner
        ? await invoiceService.getPartnerInvoice(cid, id, { cacheBust: true })
        : await getInvoice(cid, id, { cacheBust: true });
      const data = normalize(raw);
      if (!data || data.id == null) throw new Error('INVALID_INVOICE_PAYLOAD');
      return data;
    },

    async updateHeader(fields) {
      if (!partner) return null;
      const saved = await invoiceService.updatePartnerInvoice(cid, id, fields);
      return wrapInvoice(normalize(saved));
    },

    async updateLine(lineId, body) {
      if (partner) {
        const qty = body.quantity != null ? Number(body.quantity) : null;
        const lineTotal = body.line_total;
        const saved = await invoiceService.updatePartnerInvoice(cid, id, {
          lines: [
            {
              id: lineId,
              description: body.description,
              line_total: lineTotal,
              amount: lineTotal,
              adjustment_note: body.adjustment_note,
              ...(qty != null ? { quantity: qty } : {}),
            },
          ],
        });
        return wrapInvoice(normalize(saved));
      }
      return invoiceService.updateDraftInvoiceLine(cid, id, lineId, body);
    },

    async addCustomLine(body) {
      if (partner) {
        const saved = await invoiceService.updatePartnerInvoice(cid, id, {
          command: 'add_custom_line',
          description: body.description,
          line_total: body.line_total,
          qty: body.qty,
          custom_mode: body.custom_mode,
          time_unit: body.time_unit,
          service_date_iso: body.service_date_iso,
        });
        return wrapInvoice(normalize(saved));
      }
      return invoiceService.addDraftCustomLine(cid, id, body);
    },

    async removeLine(lineId, options = {}) {
      if (partner) {
        if (options.exclude_round_trip_leg) {
          const err = new Error(
            'Les factures partenaires ne gèrent pas l’exclusion d’une jambe aller-retour.'
          );
          err.response = { data: { error: err.message } };
          throw err;
        }
        const saved = await invoiceService.updatePartnerInvoice(cid, id, {
          command: 'remove_line',
          line_id: lineId,
        });
        return wrapInvoice(normalize(saved));
      }
      return invoiceService.removeDraftInvoiceLine(cid, id, lineId, options);
    },

    async applyGlobalDiscount(payload) {
      if (partner) {
        const saved = await invoiceService.updatePartnerInvoice(cid, id, {
          command: 'apply_global_discount',
          global_discount_percent: payload.global_discount_percent,
          global_discount_note: payload.global_discount_note,
        });
        return wrapInvoice(normalize(saved));
      }
      return invoiceService.applyDraftGlobalDiscount(cid, id, payload);
    },

    async applyPerLineDiscount(payload) {
      if (partner) {
        const saved = await invoiceService.updatePartnerInvoice(cid, id, {
          command: 'apply_per_line_discounts',
          line_discounts: payload.line_discounts,
        });
        return wrapInvoice(normalize(saved));
      }
      return invoiceService.applyDraftPerLineDiscounts(cid, id, payload);
    },

    async removeDiscount(options = {}) {
      if (partner) {
        const saved = await invoiceService.updatePartnerInvoice(cid, id, {
          command: 'remove_discount',
        });
        return wrapInvoice(normalize(saved));
      }
      return invoiceService.removeDraftGlobalDiscount(cid, id, options);
    },

    async regeneratePdf() {
      if (partner) {
        return invoiceService.forceRegeneratePartnerInvoicePdf(cid, id);
      }
      return invoiceService.forceRegenerateInvoicePdf(cid, id);
    },

    async sendByEmail(options = {}) {
      if (partner) {
        return invoiceService.sendPartnerInvoiceByEmail(cid, id, options);
      }
      return invoiceService.sendInvoiceByEmail(cid, id, options);
    },

    async markAsSent() {
      if (partner) {
        return invoiceService.markPartnerInvoiceAsSent(cid, id);
      }
      return invoiceService.markInvoiceAsSent(cid, id);
    },
  };
}
