/**
 * Anti-régression permanente — contrat figé « Régénérer PDF » (CLOSED).
 * Voir docs/facturation/regenerer-pdf-contrat.md.
 * Toute évolution UI doit continuer à passer par forceRegenerateInvoicePdf.
 */
import { invoiceService } from '../../../../services/invoiceService';
import apiClient from '../../../../utils/apiClient';

jest.mock('../../../../utils/apiClient', () => ({
  __esModule: true,
  default: {
    post: jest.fn(),
  },
}));

describe('forceRegenerateInvoicePdf — contrat unique', () => {
  beforeEach(() => {
    apiClient.post.mockReset();
  });

  it('appelle l’endpoint regenerate-pdf et retourne la nouvelle URL', async () => {
    apiClient.post.mockResolvedValue({
      data: {
        message: 'PDF régénéré',
        pdf_url: '/uploads/invoices/new.pdf',
        pdf_generated_at: '2026-09-16T10:00:00Z',
      },
    });

    const payload = await invoiceService.forceRegenerateInvoicePdf(3, 99);

    expect(apiClient.post).toHaveBeenCalledWith(
      expect.stringMatching(/\/invoices\/companies\/3\/invoices\/99\/regenerate-pdf$/)
    );
    expect(payload.pdf_url).toBe('/uploads/invoices/new.pdf');
  });

  it('regenerateInvoicePdf est un alias du même contrat', async () => {
    apiClient.post.mockResolvedValue({
      data: { pdf_url: '/uploads/invoices/new.pdf' },
    });

    await invoiceService.regenerateInvoicePdf(3, 99);
    await invoiceService.forceRegenerateInvoicePdf(3, 99);

    expect(apiClient.post).toHaveBeenCalledTimes(2);
    expect(apiClient.post.mock.calls[0][0]).toBe(apiClient.post.mock.calls[1][0]);
  });

  it('ne déclare jamais le succès sans nouvelle URL', async () => {
    apiClient.post.mockResolvedValue({
      data: { message: 'PDF régénéré' },
    });

    await expect(invoiceService.forceRegenerateInvoicePdf(3, 99)).rejects.toThrow(
      /nouveau document/i
    );
  });
});
