/**
 * @jest-environment jsdom
 */

import {
  labelForAuthCode,
  labelForConnectError,
  labelForDisconnectReason,
  labelForMissingToken,
} from './socketStatusReasons';

describe('socketStatusReasons', () => {
  it('mappe les codes auth', () => {
    expect(labelForAuthCode('AUTH_FORBIDDEN').reasonLabel).toMatch(/refusé/i);
    expect(labelForMissingToken().reasonCode).toBe('AUTH_REQUIRED');
  });

  it('mappe les raisons de déconnexion Socket.IO', () => {
    expect(labelForDisconnectReason('ping timeout').reasonLabel).toMatch(/attente/i);
    expect(labelForDisconnectReason('transport close').reasonCode).toBe('transport close');
  });

  it('résume les erreurs de transport', () => {
    expect(labelForConnectError('xhr poll error').reasonLabel).toMatch(/injoignable/i);
    expect(labelForConnectError('AUTH_REQUIRED').reasonCode).toBe('AUTH_REQUIRED');
  });
});
