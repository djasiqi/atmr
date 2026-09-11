import {
  fetchAuthMeSingleFlight,
  resetAuthMeSingleFlightForTests,
} from '../authMeSingleFlight';

describe('fetchAuthMeSingleFlight', () => {
  beforeEach(() => {
    resetAuthMeSingleFlightForTests();
  });

  it('partage la même promesse si un appel est déjà en vol', async () => {
    let resolveFirst;
    const requestFn = jest.fn(
      () => new Promise((resolve) => {
        resolveFirst = resolve;
      }),
    );

    const a = fetchAuthMeSingleFlight(requestFn);
    const b = fetchAuthMeSingleFlight(requestFn);
    expect(a).toBe(b);
    expect(requestFn).toHaveBeenCalledTimes(1);

    resolveFirst({ user: { id: 1 } });
    await expect(a).resolves.toEqual({ user: { id: 1 } });
    await expect(b).resolves.toEqual({ user: { id: 1 } });
  });

  it('autorise un nouvel appel une fois le précédent terminé', async () => {
    const requestFn = jest.fn().mockResolvedValueOnce('one').mockResolvedValueOnce('two');
    await expect(fetchAuthMeSingleFlight(requestFn)).resolves.toBe('one');
    await expect(fetchAuthMeSingleFlight(requestFn)).resolves.toBe('two');
    expect(requestFn).toHaveBeenCalledTimes(2);
  });
});
