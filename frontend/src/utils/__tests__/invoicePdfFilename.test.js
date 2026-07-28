import {
  buildInvoicePdfDownloadFilename,
  formatInvoiceAmountForFilename,
  resolveInvoiceFilenameClientLabel,
  slugifyInvoiceFilenamePart,
} from '../invoicePdfFilename';

describe('invoicePdfFilename', () => {
  test('slugify retire accents et caractères spéciaux', () => {
    expect(slugifyInvoiceFilenamePart('M. VUILLE Michel')).toBe('M_VUILLE_Michel');
    expect(slugifyInvoiceFilenamePart('Février')).toBe('Fevrier');
  });

  test('montant entier et décimal', () => {
    expect(formatInvoiceAmountForFilename(155)).toBe('155CHF');
    expect(formatInvoiceAmountForFilename(155.5)).toBe('155_50CHF');
  });

  test('libellé client depuis bill_to_client', () => {
    expect(
      resolveInvoiceFilenameClientLabel({
        bill_to_client: { last_name: 'VUILLE', first_name: 'Michel' },
      })
    ).toBe('VUILLE Michel');
  });

  test('nom de fichier complet', () => {
    const name = buildInvoicePdfDownloadFilename({
      period_month: 7,
      period_year: 2026,
      invoice_number: 'EM-2026-07-0005',
      total_amount: 155,
      bill_to_client: {
        username: 'M VUILLE Michel',
        last_name: '',
        first_name: '',
      },
    });
    expect(name).toBe(
      'Facture_Juillet_2026_EM-2026-07-0005_M_VUILLE_Michel_155CHF.pdf'
    );
  });
});
