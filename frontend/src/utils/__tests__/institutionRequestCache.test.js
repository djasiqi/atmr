import {
  applyOperationalBookingPatchToRequest,
  upsertInstitutionRequestInLists,
} from '../institutionRequestCache';

describe('upsertInstitutionRequestInLists', () => {
  it('insère une nouvelle demande en tête', () => {
    const next = upsertInstitutionRequestInLists(
      { requests: [{ id: 1, status: 'SENT' }], total: 1 },
      { id: 2323, status: 'DRAFT' },
    );
    expect(next.requests[0].id).toBe(2323);
    expect(next.total).toBe(2);
  });

  it('met à jour une demande existante sans changer le total', () => {
    const next = upsertInstitutionRequestInLists(
      { requests: [{ id: 2323, status: 'DRAFT' }], total: 1 },
      { id: 2323, status: 'SENT', sent_at: '2026-09-10T21:00:00' },
    );
    expect(next.requests).toHaveLength(1);
    expect(next.requests[0].status).toBe('SENT');
    expect(next.total).toBe(1);
  });
});

describe('applyOperationalBookingPatchToRequest', () => {
  it('écrit le RDV 13:00 dans les legs et invalide la confirmation de départ', () => {
    const next = applyOperationalBookingPatchToRequest(
      {
        id: 2339,
        return_to_institution: true,
        pickup_time_confirmed: true,
        booking_summary: { id: 45726, time_confirmed: true, edit_version: 1 },
        legs: [
          { sequence_index: 0, scheduled_time: '2026-09-12T14:00:00', time_confirmed: true },
          { sequence_index: 1, scheduled_time: null, is_return_stop: true },
        ],
      },
      {
        appointment_time: '2026-09-12T13:00:00',
        leg_appointments: [{ index: 0, scheduled_time: '2026-09-12T13:00:00' }],
      },
      { pickup_reconfirmation_required: true, edit_version: 2 },
    );

    expect(next.legs[0].scheduled_time).toBe('2026-09-12T13:00:00');
    expect(next.legs[0].time_confirmed).toBe(true);
    expect(next.booking_summary.time_confirmed).toBe(false);
    expect(next.booking_summary.edit_version).toBe(2);
    expect(next.pickup_time_confirmed).toBe(false);
  });
});
