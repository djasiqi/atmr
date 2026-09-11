import { shouldSkipCompanyReservationsPrefetch } from '../companyReservationsPrefetch';

describe('shouldSkipCompanyReservationsPrefetch', () => {
  it('ne skippe pas s’il n’y a pas d’état', () => {
    expect(shouldSkipCompanyReservationsPrefetch(undefined)).toBe(false);
  });

  it('skippe un fetch déjà en vol', () => {
    expect(shouldSkipCompanyReservationsPrefetch({
      fetchStatus: 'fetching',
      data: null,
      dataUpdatedAt: 0,
    })).toBe(true);
  });

  it('skippe un cache encore frais (< 60s)', () => {
    const now = 1_000_000;
    expect(shouldSkipCompanyReservationsPrefetch({
      fetchStatus: 'idle',
      data: { reservations: [{ id: 1 }] },
      dataUpdatedAt: now - 10_000,
    }, now)).toBe(true);
  });

  it('refetch si le cache est périmé', () => {
    const now = 1_000_000;
    expect(shouldSkipCompanyReservationsPrefetch({
      fetchStatus: 'idle',
      data: { reservations: [{ id: 1 }] },
      dataUpdatedAt: now - 61_000,
    }, now)).toBe(false);
  });
});
