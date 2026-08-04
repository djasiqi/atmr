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

  it('traduit missing_token au lieu du message JWT anglais', () => {
    const err = {
      response: {
        status: 401,
        data: {
          error: 'missing_token',
          message: 'Missing JWT in cookies or headers (Missing cookie "access_token")',
        },
      },
    };
    expect(getApiErrorMessage(err, 'Défaut')).toBe(
      'Session expirée. Veuillez vous reconnecter.'
    );
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

  it('explique la restriction commerciale LIRIE full', () => {
    const err = {
      message: 'Request failed with status code 403',
      response: {
        status: 403,
        data: {
          error: 'Création de course bloquée…',
          error_code: 'billing_access_restricted',
          details: {
            billing_access_state: 'full',
            capability: 'CREATE_OWN_PORTFOLIO_BOOKING',
          },
        },
      },
    };
    const msg = getApiErrorMessage(err, 'Défaut');
    expect(msg).toMatch(/Nouvelle course impossible/i);
    expect(msg).toMatch(/recouvrement/i);
    expect(msg).toMatch(/facturation plateforme/i);
    expect(msg).toMatch(/022 512 02 03/);
    expect(msg).toMatch(/info@lirie\.ch/);
    expect(msg).not.toMatch(/CREATE_OWN_PORTFOLIO/i);
  });
});
