import { buildIdentityFromApi, buildOfferIdentity, matchesSearchIndex } from '../bookingIdentity';

describe('buildIdentityFromApi', () => {
  it('lit le bloc API identity', () => {
    const booking = {
      identity: {
        passenger: { name: 'Jean Dupont' },
        source: { type: 'institution', id: 1, code: 'HUG', name: 'HUG' },
        requester: { id: 2, name: 'Marie Martin' },
        ownership: { owner_company_id: 10, owner_company_name: 'EM' },
        execution: { executing_company_id: 10, executing_company_name: 'EM' },
        upstream: null,
        origin_channel: 'institution_portal',
      },
    };
    const view = buildIdentityFromApi(booking);
    expect(view.passengerLabel).toBe('Jean Dupont');
    expect(view.source.name).toBe('HUG');
    expect(view.requester.name).toBe('Marie Martin');
  });

  it('fallback company_client sans identity', () => {
    const view = buildIdentityFromApi({
      client: { id: 5, client_type: 'TRANSPORT' },
      client_name: 'Jean Dupont',
    });
    expect(view.passengerLabel).toBe('Jean Dupont');
    expect(view.source.type).toBe('company_client');
    expect(view.source.name).toBe('Portefeuille propre');
  });
});

describe('buildOfferIdentity', () => {
  it('affiche passager puis institution', () => {
    const view = buildOfferIdentity({
      transport_request: {
        patient_name: 'Paul Patient',
        institution_name: 'Clinique LHA',
        institution_id: 3,
      },
    });
    expect(view.passengerLabel).toBe('Paul Patient');
    expect(view.source.name).toBe('Clinique LHA');
  });

  it('sans passager : institution seule en ligne 1', () => {
    const view = buildOfferIdentity({
      transport_request: { institution_name: 'HUG' },
    });
    expect(view.passengerLabel).toBe('HUG');
    expect(view.source.name).toBeNull();
  });
});

describe('matchesSearchIndex', () => {
  it('filtre via search_index serveur', () => {
    const booking = { search_index: ['Jean Dupont', 'HUG'] };
    expect(matchesSearchIndex(booking, 'hug')).toBe(true);
    expect(matchesSearchIndex(booking, 'inconnu')).toBe(false);
  });
});
