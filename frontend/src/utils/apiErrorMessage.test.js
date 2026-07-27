import { getApiErrorMessage, isServiceUnavailableError } from './apiErrorMessage';

describe('getApiErrorMessage', () => {
  it('utilise le fallback si pas de réponse', () => {
    expect(getApiErrorMessage(new Error('x'), 'Défaut')).toBe('x');
    expect(getApiErrorMessage({}, 'Défaut')).toBe('Défaut');
  });

  it('remplace les erreurs de panne/maintenance par le message support', () => {
    const err = {
      message: 'Request failed with status code 404',
      response: {
        status: 404,
        data: '404 page not found',
      },
    };
    expect(isServiceUnavailableError(err)).toBe(true);
    expect(getApiErrorMessage(err, 'Défaut')).toMatch(/momentanément indisponible/i);
    expect(getApiErrorMessage(err, 'Défaut')).toMatch(/022 512 02 03/);
    expect(getApiErrorMessage(err, 'Défaut')).toMatch(/info@lirie\.ch/);
  });

  it('détecte une panne réseau sans réponse HTTP', () => {
    const err = { code: 'ERR_NETWORK', message: 'Network Error' };
    expect(isServiceUnavailableError(err)).toBe(true);
    expect(getApiErrorMessage(err, 'Défaut')).toMatch(/contacter le support/i);
  });

  it('ignore le message Axios générique de statut HTTP', () => {
    const err = {
      message: 'Request failed with status code 503',
      response: {
        data: {
          error: 'payment_unavailable',
          message: 'Paiement Saferpay non configuré sur ce serveur',
        },
      },
    };
    expect(getApiErrorMessage(err, 'Défaut')).toBe(
      'Paiement Saferpay non configuré sur ce serveur'
    );
  });

  it('préfère message API à fallback pour erreur structurée', () => {
    const err = {
      message: 'Request failed with status code 400',
      response: { data: { message: 'Montant invalide', error: 'validation_error' } },
    };
    expect(getApiErrorMessage(err, 'Défaut')).toBe('Montant invalide');
  });

  it('utilise le message d’une Error métier (ex. après startSaferpayHostedCheckout)', () => {
    expect(
      getApiErrorMessage(new Error('URL de retour non autorisée'), 'Défaut')
    ).toBe('URL de retour non autorisée');
  });

  it('message imbriqué dans data.data', () => {
    const err = {
      response: { data: { data: { message: 'Détail interne' } } },
    };
    expect(getApiErrorMessage(err, 'Défaut')).toBe('Détail interne');
  });

  it('lit le texte dans error quand error_code est présent (format legacy)', () => {
    const err = {
      message: 'Request failed with status code 400',
      response: {
        data: {
          error: 'Incorrect old password',
          error_code: 'validation_error',
        },
      },
    };
    expect(getApiErrorMessage(err, 'Défaut')).toBe('Incorrect old password');
  });
});
