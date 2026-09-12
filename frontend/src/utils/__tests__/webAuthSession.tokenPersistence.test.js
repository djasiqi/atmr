import {
  AUTH_SECRET_STORAGE_KEYS,
  hasActiveSession,
  purgePersistedAuthSecrets,
  writeAuthSession,
} from '../webAuthSession';

const SECRET_KEYS = [
  'access_token',
  'refresh_token',
  'jwt',
  'authorization',
  'authToken',
  'refreshToken',
  'app_access_token',
  'app_refresh_token',
  'demo_access_token',
  'demo_refresh_token',
  'company_access_token',
];

describe('webAuthSession — aucune persistance JWT navigateur', () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it('writeAuthSession persiste le user mais jamais les tokens', () => {
    writeAuthSession({
      env: 'app',
      user: { public_id: 'u-1', role: 'company', email: 'c@example.com' },
      role: 'company',
    });

    expect(JSON.parse(localStorage.getItem('app_user')).public_id).toBe('u-1');
    expect(localStorage.getItem('public_id')).toBe('u-1');
    expect(hasActiveSession('app')).toBe(true);

    SECRET_KEYS.forEach((key) => {
      expect(localStorage.getItem(key)).toBeNull();
    });
    AUTH_SECRET_STORAGE_KEYS.forEach((key) => {
      expect(localStorage.getItem(key)).toBeNull();
    });
  });

  it('purge les secrets legacy au boot sans les réinjecter', () => {
    localStorage.setItem('app_access_token', 'legacy-jwt');
    localStorage.setItem('authToken', 'legacy-jwt');
    localStorage.setItem('refreshToken', 'legacy-refresh');
    localStorage.setItem('company_access_token', 'legacy-company');
    localStorage.setItem(
      'app_user',
      JSON.stringify({ public_id: 'u-legacy', role: 'company' })
    );

    purgePersistedAuthSecrets();

    expect(localStorage.getItem('app_access_token')).toBeNull();
    expect(localStorage.getItem('authToken')).toBeNull();
    expect(localStorage.getItem('refreshToken')).toBeNull();
    expect(localStorage.getItem('company_access_token')).toBeNull();
    expect(JSON.parse(localStorage.getItem('app_user')).public_id).toBe('u-legacy');
    expect(hasActiveSession('app')).toBe(true);
  });
});
