import {
  ELIGIBLE_CARRIERS_COPY,
  resolveEligibleCarriersView,
  shouldRetryEligibleCarriers,
} from '../institutionEligibleCarriers';

describe('resolveEligibleCarriersView', () => {
  it('affiche un loader uniquement sans cache', () => {
    expect(resolveEligibleCarriersView({ isPending: true })).toEqual({ kind: 'loading' });
  });

  it('distingue erreur, accès refusé, vide et succès', () => {
    expect(resolveEligibleCarriersView({
      isPending: false,
      isError: true,
      error: { response: { status: 500 } },
    })).toEqual({ kind: 'error' });
    expect(resolveEligibleCarriersView({
      isPending: false,
      isError: true,
      error: { response: { status: 403 } },
    })).toEqual({ kind: 'forbidden' });
    expect(resolveEligibleCarriersView({
      isPending: false,
      isError: false,
      availableCount: 0,
      notYetAddedCount: 0,
      catalogTotal: 0,
    })).toEqual({ kind: 'empty' });
    expect(resolveEligibleCarriersView({
      isPending: false,
      isError: false,
      availableCount: 0,
      notYetAddedCount: 0,
      catalogTotal: 4,
    })).toEqual({ kind: 'already_added' });
    expect(resolveEligibleCarriersView({
      isPending: false,
      isError: false,
      availableCount: 0,
      notYetAddedCount: 2,
      catalogTotal: 4,
      search: 'HUG',
    })).toEqual({ kind: 'search_empty' });
    expect(resolveEligibleCarriersView({
      isPending: false,
      isError: false,
      availableCount: 3,
      notYetAddedCount: 3,
      catalogTotal: 3,
    })).toEqual({ kind: 'success' });
  });

  it('ne relance pas 401/403 ni un abort', () => {
    expect(shouldRetryEligibleCarriers(0, { response: { status: 401 } })).toBe(false);
    expect(shouldRetryEligibleCarriers(0, { response: { status: 403 } })).toBe(false);
    expect(shouldRetryEligibleCarriers(0, { code: 'ERR_CANCELED' })).toBe(false);
    expect(shouldRetryEligibleCarriers(0, { response: { status: 500 } })).toBe(true);
    expect(shouldRetryEligibleCarriers(1, { response: { status: 500 } })).toBe(false);
  });

  it('a un libellé de retry', () => {
    expect(ELIGIBLE_CARRIERS_COPY.retry).toBe('Réessayer');
  });
});
