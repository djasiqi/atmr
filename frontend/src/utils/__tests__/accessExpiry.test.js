/**
 * accessExpiry : métadonnée scheduler sans JWT.
 */

const {
  resolveAccessExpiresAtMs,
  noteAccessExpiryFromResponse,
  getStoredAccessExpiresAtMs,
  clearStoredAccessExpiry,
  isAccessNearExpiry,
  isAccessExpired,
  ACCESS_EXPIRES_AT_STORAGE_KEY,
} = require('../accessExpiry');

describe('accessExpiry', () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it('résout access_expires_at ISO et access_expires_in', () => {
    const now = Date.parse('2026-08-07T08:00:00.000Z');
    expect(resolveAccessExpiresAtMs({ access_expires_in: 3600 }, now)).toBe(now + 3600_000);
    expect(
      resolveAccessExpiresAtMs({ access_expires_at: '2026-08-07T09:00:00.000Z' }, now)
    ).toBe(Date.parse('2026-08-07T09:00:00.000Z'));
  });

  it('persiste sans JWT dans localStorage', () => {
    const now = Date.now();
    noteAccessExpiryFromResponse({ access_expires_in: 3600 }, now);
    expect(localStorage.getItem(ACCESS_EXPIRES_AT_STORAGE_KEY)).toBeTruthy();
    expect(localStorage.getItem('app_access_token')).toBeNull();
    expect(getStoredAccessExpiresAtMs()).toBeGreaterThan(now);
    clearStoredAccessExpiry();
    expect(getStoredAccessExpiresAtMs()).toBeNull();
  });

  it('détecte near expiry / expired', () => {
    const now = Date.now();
    noteAccessExpiryFromResponse({ access_expires_in: 60 }, now);
    expect(isAccessNearExpiry(5 * 60 * 1000, now)).toBe(true);
    expect(isAccessExpired(now)).toBe(false);
    expect(isAccessExpired(now + 120_000)).toBe(true);
  });
});
