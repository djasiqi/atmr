import {
  downloadPortalTermsDocument,
  portalTermsDocumentFilename,
  portalTermsDocumentLabel,
  portalTermsPdfApiPath,
  printPortalTermsDocument,
} from '../portalTermsDocument';
import { downloadProtectedPdfAsFile } from '../protectedPdf';

jest.mock('../protectedPdf', () => ({
  downloadProtectedPdfAsFile: jest.fn(async () => true),
}));

describe('portalTermsDocument', () => {
  it('libellé selon le type et la variante', () => {
    expect(portalTermsDocumentLabel('terms_of_service')).toContain('utilisation');
    expect(portalTermsDocumentLabel('transport_terms')).toContain('réservation');
    expect(portalTermsDocumentLabel('transport_terms', { variant: 'order' })).toContain(
      'transport'
    );
  });

  it('construit un nom de fichier PDF officiel', () => {
    expect(
      portalTermsDocumentFilename({
        document_type: 'transport_terms',
        current_version: '2.0',
      })
    ).toBe('LIRIE_CGV_reservation_transport_v2.0.pdf');
    expect(
      portalTermsPdfApiPath({ document_type: 'terms_of_service' })
    ).toBe('/clients/me/portal-terms/terms_of_service/pdf');
  });

  it('télécharge le PDF officiel via l’API', async () => {
    downloadProtectedPdfAsFile.mockResolvedValue(true);
    const ok = await downloadPortalTermsDocument({
      document_type: 'terms_of_service',
      current_version: '2.0',
      canonical_body: 'CORPS CGU 2.0',
    });
    expect(downloadProtectedPdfAsFile).toHaveBeenCalledWith(
      '/clients/me/portal-terms/terms_of_service/pdf',
      'LIRIE_CGU_compte_client_prive_v2.0.pdf'
    );
    expect(ok).toBe(true);
  });

  it('ouvre une fenêtre d’impression avec logo', () => {
    const write = jest.fn();
    const close = jest.fn();
    const focus = jest.fn();
    const print = jest.fn();
    window.open = jest.fn(() => ({
      document: { open: jest.fn(), write, close },
      focus,
      print,
    }));
    jest.useFakeTimers();

    const ok = printPortalTermsDocument({
      document_type: 'transport_terms',
      current_version: '2.0',
      canonical_body: 'CORPS CGV 2.0',
    });

    expect(ok).toBe(true);
    expect(window.open).toHaveBeenCalled();
    expect(write.mock.calls[0][0]).toContain('logo-lirie.png');
    expect(write.mock.calls[0][0]).toContain('CORPS CGV 2.0');
    jest.runAllTimers();
    expect(print).toHaveBeenCalled();
    jest.useRealTimers();
  });
});
