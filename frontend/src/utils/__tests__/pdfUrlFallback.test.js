import {
  buildInvoicePdfApiUrl,
  buildPartnerInvoicePdfApiUrl,
  isPartnerInvoice,
  resolveInvoicePdfApiUrl,
} from '../pdfUrlFallback';

describe('catalogues facture client vs partenaire', () => {
  const companyId = 1;
  const sharedId = 34;

  const standardInvoice = {
    id: sharedId,
    company_id: companyId,
    invoice_number: 'EM-2026-08-0034',
  };

  const partnerInvoice = {
    id: sharedId,
    company_id: companyId,
    invoice_number: 'PARTNER-EM-2026-08-0097',
    is_partner_invoice: true,
  };

  test('isPartnerInvoice ne se fie qu’au flag catalogue', () => {
    expect(isPartnerInvoice(standardInvoice)).toBe(false);
    expect(isPartnerInvoice(partnerInvoice)).toBe(true);
    expect(isPartnerInvoice({ id: sharedId })).toBe(false);
  });

  test('IDs identiques : les builders ne se mélangent pas', () => {
    expect(buildInvoicePdfApiUrl(standardInvoice)).toBe(
      `/invoices/companies/${companyId}/invoices/${sharedId}/pdf`
    );
    expect(buildPartnerInvoicePdfApiUrl(partnerInvoice)).toBe(
      `/invoices/companies/${companyId}/partner-invoices/${sharedId}/pdf`
    );
    expect(buildInvoicePdfApiUrl(standardInvoice)).not.toBe(
      buildPartnerInvoicePdfApiUrl(partnerInvoice)
    );
  });

  test('resolveInvoicePdfApiUrl route le partenaire hors de /invoices/{id}', () => {
    expect(resolveInvoicePdfApiUrl(standardInvoice)).toBe(
      `/invoices/companies/${companyId}/invoices/${sharedId}/pdf`
    );
    expect(resolveInvoicePdfApiUrl(partnerInvoice)).toBe(
      `/invoices/companies/${companyId}/partner-invoices/${sharedId}/pdf`
    );
    expect(resolveInvoicePdfApiUrl(partnerInvoice)).not.toMatch(
      `/invoices/${sharedId}`
    );
  });

  test('company_id fourni à part suffit pour le partenaire', () => {
    expect(
      resolveInvoicePdfApiUrl(
        { id: sharedId, is_partner_invoice: true },
        companyId
      )
    ).toBe(`/invoices/companies/${companyId}/partner-invoices/${sharedId}/pdf`);
  });
});
