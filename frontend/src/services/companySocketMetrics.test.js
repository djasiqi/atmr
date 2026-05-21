/**
 * @jest-environment jsdom
 */

import {
  MAX_DEV_LISTENERS_PER_EVENT,
  setActiveListenerCount,
  trackListenerCountForWatchdog,
  warnIfTooManyListeners,
} from './companySocketMetrics';

describe('companySocketMetrics', () => {
  const originalEnv = process.env.NODE_ENV;

  afterEach(() => {
    process.env.NODE_ENV = originalEnv;
    jest.restoreAllMocks();
  });

  it('warnIfTooManyListeners is silent in production', () => {
    process.env.NODE_ENV = 'production';
    const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
    warnIfTooManyListeners('driver_location_update', MAX_DEV_LISTENERS_PER_EVENT + 5);
    expect(warnSpy).not.toHaveBeenCalled();
  });

  it('warnIfTooManyListeners warns in development above threshold', () => {
    process.env.NODE_ENV = 'development';
    const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
    warnIfTooManyListeners('booking_updated', MAX_DEV_LISTENERS_PER_EVENT + 1);
    expect(warnSpy).toHaveBeenCalled();
  });

  it('trackListenerCountForWatchdog warns on sustained growth in dev', () => {
    process.env.NODE_ENV = 'development';
    const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
    setActiveListenerCount(2);
    trackListenerCountForWatchdog(2);
    trackListenerCountForWatchdog(12);
    expect(warnSpy).toHaveBeenCalled();
  });
});
