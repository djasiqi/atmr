import { formatServerAmount, loadInvoiceSituation, readStatement } from './invoiceSituation';

describe('consultation de la situation', () => {
  it('affiche la chaîne du serveur et laisse un montant absent inconnu', () => {
    expect(formatServerAmount('60.00')).toBe('60.00 CHF');
    expect(formatServerAmount('-30.00')).toBe('-30.00 CHF');
    expect(formatServerAmount(null)).toBe('Montant non indiqué');
    expect(formatServerAmount('')).toBe('Montant non indiqué');
  });

  it('lit la situation sans la recalculer', () => {
    const statement = readStatement({
      data: {
        total_amount: 90,
        statement: { net_facture_chf: '60.00', disponible: true },
      },
    });
    expect(statement.net_facture_chf).toBe('60.00');
  });

  it('ne consulte qu une facture existante', async () => {
    const invoiceService = {
      getInvoice: jest.fn(async () => ({
        data: { statement: { disponible: true, reste_a_payer_chf: '40.00' } },
      })),
      getPartnerInvoice: jest.fn(),
      postPaymentForResource: jest.fn(),
      forceRegenerateInvoicePdfForResource: jest.fn(),
    };
    const result = await loadInvoiceSituation({
      invoiceService,
      companyId: 4,
      invoice: { id: 9, invoice_type: 'standard' },
    });
    expect(invoiceService.getInvoice).toHaveBeenCalledWith(4, 9);
    expect(invoiceService.getPartnerInvoice).not.toHaveBeenCalled();
    expect(invoiceService.postPaymentForResource).not.toHaveBeenCalled();
    expect(invoiceService.forceRegenerateInvoicePdfForResource).not.toHaveBeenCalled();
    expect(result.statement.reste_a_payer_chf).toBe('40.00');
  });

  it('garde le catalogue partenaire sur sa propre consultation', async () => {
    const invoiceService = {
      getInvoice: jest.fn(),
      getPartnerInvoice: jest.fn(async () => ({
        data: {
          statement: {
            disponible: true,
            prestations_chf: '80.00',
            avoirs_chf: null,
            reste_a_payer_chf: '80.00',
          },
        },
      })),
    };
    const result = await loadInvoiceSituation({
      invoiceService,
      companyId: 4,
      invoice: { id: 3, is_partner_invoice: true },
    });
    expect(invoiceService.getInvoice).not.toHaveBeenCalled();
    expect(invoiceService.getPartnerInvoice).toHaveBeenCalledWith(4, 3);
    expect(result.statement.avoirs_chf).toBeNull();
  });

  it('remonte le refus et le document incoherent', async () => {
    const refused = await loadInvoiceSituation({
      invoiceService: {
        getInvoice: async () => ({
          data: {
            statement: {
              disponible: false,
              message: 'Entreprise non trouvée ou accès refusé',
            },
          },
        }),
      },
      companyId: 4,
      invoice: { id: 9 },
    });
    expect(refused.statement).toBeNull();
    expect(refused.error).toMatch(/refus/i);
  });
});
