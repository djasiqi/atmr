import {
  SEND_RETRY_TOAST,
  institutionSubmitBusy,
  institutionSubmitButtonLabel,
  resolveCreatedRequestId,
} from '../institutionRequestSubmit';

describe('institutionRequestSubmit', () => {
  it('réutilise l’id déjà créé pour ne pas recréer', () => {
    expect(resolveCreatedRequestId(2323, { id: 9999 })).toBe(2323);
    expect(resolveCreatedRequestId(null, { id: 2323 })).toBe(2323);
    expect(resolveCreatedRequestId(null, null)).toBeNull();
  });

  it('détecte un submit en cours', () => {
    expect(institutionSubmitBusy('send', {})).toBe(true);
    expect(institutionSubmitBusy(null, { createPending: true })).toBe(true);
    expect(institutionSubmitBusy(null, {})).toBe(false);
  });

  it('affiche Envoi… puis réessai sans recréer', () => {
    expect(institutionSubmitButtonLabel({
      isLirieSendMode: true,
      busy: true,
    })).toBe('Envoi…');
    expect(institutionSubmitButtonLabel({
      isLirieSendMode: true,
      busy: false,
      hasCreatedRequest: true,
    })).toBe("Réessayer l'envoi LIRIE");
    expect(SEND_RETRY_TOAST).toMatch(/ne sera pas recréée/);
  });
});
