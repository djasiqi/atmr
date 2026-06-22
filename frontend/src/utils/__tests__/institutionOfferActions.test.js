import { resolveInstitutionOfferActions } from '../institutionOfferActions';

const NOW = new Date('2026-06-22T22:21:00+02:00');

const pendingOffer = (req) => ({
  id: 1,
  status: 'PENDING',
  can_respond: true,
  expires_at: '2030-01-01T12:00:00Z',
  transport_request: req,
});

describe('resolveInstitutionOfferActions', () => {
  it('masque toute action si offre expirée', () => {
    const actions = resolveInstitutionOfferActions(
      {
        id: 1,
        status: 'PENDING',
        can_respond: false,
        expires_at: '2020-01-01T12:00:00Z',
        transport_request: { is_urgent: true },
      },
      NOW
    );
    expect(actions.canRespond).toBe(false);
    expect(actions.canAcceptNow).toBe(false);
  });

  it('cas Khalid : Planifier + Refuser uniquement', () => {
    const actions = resolveInstitutionOfferActions(
      pendingOffer({
        pickup_time_confirmed: false,
        scheduled_time: '2026-06-22T20:00:00',
        scheduled_time_type: 'arrival',
        appointment_time_confirmed: true,
        is_urgent: false,
        legs: [{ sequence_index: 0, scheduled_time: '2026-06-22T20:00:00', time_confirmed: true }],
      }),
      NOW
    );
    expect(actions.canValidate).toBe(false);
    expect(actions.canAcceptNow).toBe(false);
    expect(actions.canPlan).toBe(true);
    expect(actions.canReject).toBe(true);
  });
});
