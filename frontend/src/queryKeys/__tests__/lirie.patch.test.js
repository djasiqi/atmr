import { lirieKeys, liriePatchCompanyReservationLists } from '../lirie';

describe('liriePatchCompanyReservationLists', () => {
  it('met à jour uniquement la réservation ciblée', () => {
    const calls = [];
    const queryClient = {
      setQueriesData: (filter, updater) => {
        calls.push(filter.queryKey);
        const next = updater({
          reservations: [
            { id: 10, status: 'pending' },
            { id: 11, status: 'accepted' },
          ],
          total: 2,
        });
        expect(next.reservations[0]).toEqual({ id: 10, status: 'accepted' });
        expect(next.reservations[1]).toEqual({ id: 11, status: 'accepted' });
        expect(next.total).toBe(2);
      },
    };

    liriePatchCompanyReservationLists(queryClient, 10, { status: 'accepted' });
    expect(calls).toContainEqual(lirieKeys.companyReservationsPaginated('me', {}).slice(0, 2));
  });
});
