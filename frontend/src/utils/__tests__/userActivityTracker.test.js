import {
  applyRemoteUserActivity,
  getUserActivityStorageKey,
  onUserActivity,
  recordUserActivity,
  resetUserActivityTrackerForTests,
  startUserActivityTracking,
} from '../userActivityTracker';

jest.mock('../webAuthSession', () => ({
  getAuthEnv: jest.fn(() => 'app'),
  getActivePublicId: jest.fn(() => 'user-1'),
}));

describe('userActivityTracker cross-tab', () => {
  beforeEach(() => {
    jest.useFakeTimers();
    resetUserActivityTrackerForTests();
    localStorage.clear();
  });

  afterEach(() => {
    resetUserActivityTrackerForTests();
    jest.useRealTimers();
    localStorage.clear();
  });

  it('notifie source local sur recordUserActivity', () => {
    const listener = jest.fn();
    onUserActivity(listener);
    recordUserActivity();
    expect(listener).toHaveBeenCalledWith(expect.any(Number), { source: 'local' });
  });

  it('écrit une clé scopée env+publicId', () => {
    recordUserActivity();
    expect(localStorage.getItem(getUserActivityStorageKey())).toBeTruthy();
  });

  it('applique une activité distante valide avec source remote', () => {
    const listener = jest.fn();
    onUserActivity(listener);
    jest.advanceTimersByTime(1000);
    const ok = applyRemoteUserActivity(Date.now());
    expect(ok).toBe(true);
    expect(listener).toHaveBeenCalledWith(expect.any(Number), { source: 'remote' });
  });

  it('ignore timestamp distant invalide ou futur', () => {
    expect(applyRemoteUserActivity('abc')).toBe(false);
    expect(applyRemoteUserActivity(Date.now() + 60_000)).toBe(false);
  });

  it('ignore une clé storage d’un autre public_id', () => {
    const cleanup = startUserActivityTracking();
    const listener = jest.fn();
    onUserActivity(listener);
    window.dispatchEvent(
      new StorageEvent('storage', {
        key: 'lirie_last_user_activity:app:other-user',
        newValue: String(Date.now()),
        storageArea: localStorage,
      })
    );
    expect(listener).not.toHaveBeenCalled();
    cleanup();
  });
});
