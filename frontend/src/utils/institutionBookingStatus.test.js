import {
  computeInstitutionRequestStats,
  getInstitutionRequestKpiBucket,
  resolveBookingStatusKey,
} from './institutionBookingStatus';

describe('resolveBookingStatusKey', () => {
  it('normalise CANCELLED en CANCELED', () => {
    expect(resolveBookingStatusKey({ status: 'CANCELLED' })).toBe('CANCELED');
  });

  it('utilise overall_status pour un A/R terminé', () => {
    expect(
      resolveBookingStatusKey({
        status: 'IN_PROGRESS',
        overall_status: 'completed',
        return_booking: { status: 'COMPLETED' },
      }),
    ).toBe('RETURN_COMPLETED');
  });
});

describe('getInstitutionRequestKpiBucket', () => {
  it('classe une demande CONVERTED avec booking annulé', () => {
    expect(
      getInstitutionRequestKpiBucket({
        status: 'CONVERTED',
        booking_id: 1,
        booking_summary: { status: 'CANCELED' },
      }),
    ).toBe('cancelled');
  });

  it('classe une demande CONVERTED terminée (RETURN_COMPLETED)', () => {
    expect(
      getInstitutionRequestKpiBucket({
        status: 'CONVERTED',
        booking_id: 1,
        booking_summary: {
          status: 'COMPLETED',
          overall_status: 'completed',
          return_booking: { status: 'COMPLETED' },
        },
      }),
    ).toBe('completed');
  });

  it('classe une demande CONVERTED en cours', () => {
    expect(
      getInstitutionRequestKpiBucket({
        status: 'CONVERTED',
        booking_id: 1,
        booking_summary: { status: 'ACCEPTED' },
      }),
    ).toBe('active');
  });
});

describe('computeInstitutionRequestStats', () => {
  it('répartit total = en cours + terminés + annulés + en attente', () => {
    const items = [
      { status: 'CONVERTED', booking_id: 1, booking_summary: { status: 'ACCEPTED' } },
      { status: 'CONVERTED', booking_id: 2, booking_summary: { status: 'COMPLETED', completed_at: '2026-05-27' } },
      { status: 'CONVERTED', booking_id: 3, booking_summary: { status: 'COMPLETED', completed_at: '2026-05-27' } },
      { status: 'CONVERTED', booking_id: 4, booking_summary: { status: 'COMPLETED', completed_at: '2026-05-27' } },
      { status: 'CONVERTED', booking_id: 5, booking_summary: { status: 'CANCELED' } },
    ];

    const stats = computeInstitutionRequestStats(items, 5);

    expect(stats).toMatchObject({
      total: 5,
      active: 1,
      completed: 3,
      cancelled: 1,
      pending: 0,
    });
  });
});
