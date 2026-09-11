import { QueryClient } from '@tanstack/react-query';
import { lirieKeys } from '../../queryKeys/lirie';
import {
  ACCEPT_OFFER_ERROR_TOAST,
  REJECT_OFFER_ERROR_TOAST,
  bookingFromAcceptedOffer,
  countVisibleInstitutionOffers,
  isSupersededByLocalDecision,
  reconcileInstitutionOffersResponse,
  rememberLocalOfferDecision,
  removeInstitutionOfferFromCache,
  resetLocalInstitutionOfferDecisions,
  restoreInstitutionOfferInCache,
  tempBookingIdFromOffer,
  upsertPendingInstitutionOffer,
} from '../institutionOffersCache';

const pending = (id, extras = {}) => ({
  id,
  status: 'PENDING',
  can_respond: true,
  expires_at: '2030-01-01T12:00:00Z',
  ...extras,
});

describe('institutionOffersCache', () => {
  beforeEach(() => {
    resetLocalInstitutionOfferDecisions();
  });

  it('retire une offre et décrémente le total (badge = même liste)', () => {
    const client = new QueryClient();
    client.setQueryData(lirieKeys.institutionOffers(), {
      offers: [pending(10), pending(11), pending(12)],
      total: 3,
    });

    const { removed, index } = removeInstitutionOfferFromCache(client, 11);
    const next = client.getQueryData(lirieKeys.institutionOffers());
    expect(removed.id).toBe(11);
    expect(index).toBe(1);
    expect(next.offers.map((o) => o.id)).toEqual([10, 12]);
    expect(next.total).toBe(2);
    expect(countVisibleInstitutionOffers(next.offers)).toBe(2);
  });

  it('restaure l\'offre à son index après un échec', () => {
    const client = new QueryClient();
    client.setQueryData(lirieKeys.institutionOffers(), {
      offers: [pending(10), pending(12)],
      total: 2,
    });
    restoreInstitutionOfferInCache(client, pending(11), 1);
    const next = client.getQueryData(lirieKeys.institutionOffers());
    expect(next.offers.map((o) => o.id)).toEqual([10, 11, 12]);
  });

  it('ignore un événement PENDING après un refus local (anti-résurrection)', () => {
    rememberLocalOfferDecision(42, 'REJECTED', 100);
    expect(isSupersededByLocalDecision(42, 'PENDING', 50)).toBe(true);
    const reconciled = reconcileInstitutionOffersResponse({
      offers: [pending(42), pending(7)],
      total: 2,
    });
    expect(reconciled.offers.map((o) => o.id)).toEqual([7]);
    expect(reconciled.total).toBe(1);
  });

  it('n\'insère pas une offre refusée localement via upsert', () => {
    const client = new QueryClient();
    client.setQueryData(lirieKeys.institutionOffers(), { offers: [], total: 0 });
    rememberLocalOfferDecision(9, 'REJECTED');
    upsertPendingInstitutionOffer(client, pending(9));
    expect(client.getQueryData(lirieKeys.institutionOffers()).offers).toEqual([]);
  });

  it('ajoute une nouvelle offre PENDING (arrivée temps réel)', () => {
    const client = new QueryClient();
    client.setQueryData(lirieKeys.institutionOffers(), {
      offers: [pending(1)],
      total: 1,
    });
    upsertPendingInstitutionOffer(client, pending(2));
    const next = client.getQueryData(lirieKeys.institutionOffers());
    expect(next.offers.map((o) => o.id)).toEqual([2, 1]);
    expect(countVisibleInstitutionOffers(next.offers)).toBe(2);
  });

  it('construit une réservation locale depuis l\'offre acceptée', () => {
    const booking = bookingFromAcceptedOffer(
      pending(5, {
        transport_request: {
          patient_name: 'CAVADINI Charlotte',
          pickup_location: 'Anières',
          dropoff_location: 'HUG',
          scheduled_time: '2026-09-11T14:00:00',
          mission_date: '2026-09-11',
          institution_name: 'LHA',
        },
      }),
      { proposedPickupTime: '2026-09-11T13:30:00' },
    );
    expect(booking.id).toBe(tempBookingIdFromOffer(5));
    expect(booking.status).toBe('accepted');
    expect(booking.driver_id).toBeNull();
    expect(booking.scheduled_time).toBe('2026-09-11T13:30:00');
    expect(booking.__fromInstitutionOfferId).toBe(5);
  });

  it('expose les libellés d\'erreur attendus', () => {
    expect(REJECT_OFFER_ERROR_TOAST).toBe(
      "Le refus n'a pas pu être enregistré. Réessayez.",
    );
    expect(ACCEPT_OFFER_ERROR_TOAST).toContain('acceptation');
  });
});
