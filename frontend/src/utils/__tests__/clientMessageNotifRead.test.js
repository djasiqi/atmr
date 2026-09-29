/**
 * @jest-environment jsdom
 */
import {
  clientNotificationReadKey,
  readClientMessageIds,
  rememberClientMessageRead,
} from '../clientMessageNotifRead';

describe('clientNotificationReadKey', () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  it('garde les messages en nombre et les factures en clé distincte', () => {
    expect(clientNotificationReadKey(12)).toBe(12);
    expect(clientNotificationReadKey('invoice-2318')).toBe('invoice-2318');
    rememberClientMessageRead([12, 'invoice-2318']);
    const read = readClientMessageIds();
    expect(read.has(12)).toBe(true);
    expect(read.has('invoice-2318')).toBe(true);
    expect(read.has(2318)).toBe(false);
  });
});
