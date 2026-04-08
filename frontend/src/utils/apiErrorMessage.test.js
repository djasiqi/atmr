import { getApiErrorMessage } from './apiErrorMessage';

describe('getApiErrorMessage', () => {
  it('utilise le fallback si pas de réponse', () => {
    expect(getApiErrorMessage(new Error('x'), 'Défaut')).toBe('x');
    expect(getApiErrorMessage({}, 'Défaut')).toBe('Défaut');
  });

  it('ignore le message Axios générique de statut HTTP', () => {
    const err = {
      message: 'Request failed with status code 503',
      response: {
        data: {
          error: 'payment_unavailable',
          message: 'Paiement Worldline non configuré sur ce serveur',
        },
      },
    };
    expect(getApiErrorMessage(err, 'Défaut')).toBe(
      'Paiement Worldline non configuré sur ce serveur'
    );
  });

  it('préfère message API à fallback pour erreur structurée', () => {
    const err = {
      message: 'Request failed with status code 400',
      response: { data: { message: 'Montant invalide', error: 'validation_error' } },
    };
    expect(getApiErrorMessage(err, 'Défaut')).toBe('Montant invalide');
  });

  it('utilise le message d’une Error métier (ex. après startWorldlineHostedCheckout)', () => {
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
});
