import {
  INVOICE_CATALOG,
  clearInvoiceCatalogSearchParams,
  createInvoiceLoadSession,
  existingInvoiceOpenPlan,
  invoiceCatalogFingerprint,
  invoiceCatalogRowKey,
  readInvoiceCatalogFromSearch,
  resolveInvoiceCatalogType,
  resolveInvoiceResource,
  stampInvoiceCatalog,
  writeInvoiceCatalogSearchParams,
} from '../invoiceCatalog';

describe('invoiceCatalog — identité typée', () => {
  const companyId = 1;
  const sharedId = 34;

  const partnerInvoice = {
    id: sharedId,
    company_id: companyId,
    invoice_number: 'PARTNER-EM-2026-08-0097',
  };

  const standardInvoice = {
    id: sharedId,
    company_id: companyId,
    invoice_number: 'EM-2026-08-0034',
  };

  test('le type voyage avec l’id : 34 partenaire ≠ 34 standard', () => {
    expect(resolveInvoiceCatalogType(partnerInvoice)).toBe(INVOICE_CATALOG.PARTNER);
    expect(resolveInvoiceCatalogType(standardInvoice)).toBe(INVOICE_CATALOG.STANDARD);
    expect(invoiceCatalogFingerprint(partnerInvoice)).not.toBe(
      invoiceCatalogFingerprint(standardInvoice)
    );
    expect(invoiceCatalogRowKey(partnerInvoice)).toBe(`partner:${sharedId}`);
    expect(invoiceCatalogRowKey(standardInvoice)).toBe(`standard:${sharedId}`);
  });

  test('invoice_type=partner suffit même sans numéro', () => {
    expect(
      resolveInvoiceCatalogType({ id: sharedId }, { invoice_type: 'partner' })
    ).toBe(INVOICE_CATALOG.PARTNER);
  });

  test('un numéro PARTNER- l’emporte sur un hint standard erroné', () => {
    expect(
      resolveInvoiceCatalogType(partnerInvoice, { type: 'standard' })
    ).toBe(INVOICE_CATALOG.PARTNER);
  });

  test('resolver unique : aucun GET /invoices/{id} pour un partenaire', () => {
    const partner = resolveInvoiceResource(partnerInvoice, companyId);
    const standard = resolveInvoiceResource(standardInvoice, companyId);
    expect(partner.allowsStandardDetailGet).toBe(false);
    expect(standard.allowsStandardDetailGet).toBe(true);
    expect(partner.pdfApiUrl).toBe(
      `/invoices/companies/${companyId}/partner-invoices/${sharedId}/pdf`
    );
    expect(partner.pdfApiUrl).not.toMatch(`/invoices/${sharedId}`);
    expect(standard.pdfApiUrl).toBe(
      `/invoices/companies/${companyId}/invoices/${sharedId}/pdf`
    );
    expect(partner.detailApiUrl).toBe(
      `/invoices/companies/${companyId}/partner-invoices/${sharedId}`
    );
    expect(partner.regeneratePdfApiUrl).toBe(
      `/invoices/companies/${companyId}/partner-invoices/${sharedId}/regenerate-pdf`
    );
    expect(partner.cancelApiUrl).toBe(
      `/invoices/companies/${companyId}/partner-invoices/${sharedId}/cancel`
    );
    expect(partner.detailApiUrl).not.toMatch(`/invoices/${sharedId}`);
    expect(standard.regeneratePdfApiUrl).toBe(
      `/invoices/companies/${companyId}/invoices/${sharedId}/regenerate-pdf`
    );
  });

  test('query invoice_type + invoice_id reconstitue le catalogue', () => {
    const params = new URLSearchParams();
    writeInvoiceCatalogSearchParams(params, {
      type: INVOICE_CATALOG.PARTNER,
      id: sharedId,
    });
    expect(params.get('invoice_type')).toBe('partner');
    expect(readInvoiceCatalogFromSearch(params)).toEqual({
      type: INVOICE_CATALOG.PARTNER,
      id: sharedId,
    });
    clearInvoiceCatalogSearchParams(params);
    expect(readInvoiceCatalogFromSearch(params)).toBeNull();
  });

  test('409 PARTNER- n’autorise pas le détail catalogue client', () => {
    const plan = existingInvoiceOpenPlan({
      existingInvoiceId: sharedId,
      existingInvoiceNumber: 'PARTNER-EM-2026-08-0097',
      companyId,
    });
    expect(plan.fetchStandardDetail).toBe(false);
    expect(plan.searchParams.invoice_type).toBe('partner');
    expect(plan.resource.pdfApiUrl).not.toMatch(`/invoices/${sharedId}`);
  });

  test('session de chargement : un 404 standard périmé est ignoré', () => {
    const session = createInvoiceLoadSession();
    const first = session.begin();
    session.invalidate();
    expect(session.isCurrent(first)).toBe(false);
    const second = session.begin();
    expect(session.isCurrent(second)).toBe(true);
  });

  test('stamp partenaire pose kind + invoice_type', () => {
    const stamped = stampInvoiceCatalog({ id: sharedId }, INVOICE_CATALOG.PARTNER);
    expect(stamped.is_partner_invoice).toBe(true);
    expect(stamped.kind).toBe('partner');
    expect(stamped.invoice_type).toBe('partner');
  });
});
