import {
  recordUserActivity,
  getMsSinceLastUserActivity,
  isUserRecentlyActive,
  USER_ACTIVE_WINDOW_MS,
  startUserActivityTracking,
} from '../userActivityTracker';

describe('userActivityTracker', () => {
  beforeEach(() => {
    jest.useFakeTimers();
    recordUserActivity();
    jest.advanceTimersByTime(5000);
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it('considère l’utilisateur actif juste après une interaction', () => {
    recordUserActivity();
    expect(isUserRecentlyActive(USER_ACTIVE_WINDOW_MS)).toBe(true);
    expect(getMsSinceLastUserActivity()).toBeLessThan(2000);
  });

  it('considère l’utilisateur inactif après la fenêtre d’activité', () => {
    recordUserActivity();
    jest.advanceTimersByTime(USER_ACTIVE_WINDOW_MS + 1000);
    expect(isUserRecentlyActive(USER_ACTIVE_WINDOW_MS)).toBe(false);
  });

  it('démarre le suivi des événements DOM', () => {
    const addSpy = jest.spyOn(window, 'addEventListener');
    const cleanup = startUserActivityTracking();
    expect(addSpy).toHaveBeenCalledWith('click', expect.any(Function), expect.any(Object));
    cleanup();
    addSpy.mockRestore();
  });
});
