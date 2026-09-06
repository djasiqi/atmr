/**
 * /company_dispatch/* : session cookie httpOnly entreprise doit passer.
 * Un chauffeur sans session entreprise reste bloqué.
 */

const {
  default: apiClient,
  COMPANY_DISPATCH_MISSING_TOKEN,
} = require('../apiClient');

describe('apiClient — company_dispatch session', () => {
  let previousAdapter;

  beforeEach(() => {
    localStorage.clear();
    localStorage.setItem('lirie_auth_env', 'app');
    previousAdapter = apiClient.defaults.adapter;
  });

  afterEach(() => {
    apiClient.defaults.adapter = previousAdapter;
    localStorage.clear();
  });

  it('laisse passer GET mode en session entreprise cookie-only', async () => {
    localStorage.setItem(
      'app_user',
      JSON.stringify({ role: 'company', public_id: 'c-1' })
    );

    let capturedAuth;
    apiClient.defaults.adapter = (config) => {
      capturedAuth = config.headers?.Authorization;
      return Promise.resolve({
        data: { dispatch_mode: 'manual' },
        status: 200,
        statusText: 'OK',
        headers: {},
        config,
      });
    };

    const { data } = await apiClient.get('/company_dispatch/mode');
    expect(data.dispatch_mode).toBe('manual');
    expect(capturedAuth).toBeFalsy();
  });

  it('laisse passer PUT mode=manual en session cookie-only', async () => {
    localStorage.setItem(
      'app_user',
      JSON.stringify({ role: 'company', public_id: 'c-1' })
    );

    apiClient.defaults.adapter = (config) => {
      const url = String(config.url || '');
      if (url.includes('/auth/csrf-token')) {
        return Promise.resolve({
          data: { csrf_token: 'test-csrf', ttl: 3600 },
          status: 200,
          statusText: 'OK',
          headers: {},
          config,
        });
      }
      return Promise.resolve({
        data: { dispatch_mode: 'manual' },
        status: 200,
        statusText: 'OK',
        headers: {},
        config,
      });
    };

    const { data } = await apiClient.put('/company_dispatch/mode', {
      dispatch_mode: 'manual',
    });
    expect(data.dispatch_mode).toBe('manual');
  });

  it('refuse un chauffeur sans session entreprise', async () => {
    localStorage.setItem(
      'app_user',
      JSON.stringify({ role: 'driver', public_id: 'd-1' })
    );

    await expect(apiClient.get('/company_dispatch/mode')).rejects.toThrow(
      COMPANY_DISPATCH_MISSING_TOKEN
    );
  });

  it('attache le Bearer si un JWT entreprise JS est présent', async () => {
    localStorage.setItem('company_access_token', 'jwt-company');
    localStorage.setItem(
      'app_user',
      JSON.stringify({ role: 'company', public_id: 'c-1' })
    );

    let capturedAuth;
    apiClient.defaults.adapter = (config) => {
      capturedAuth = config.headers?.Authorization;
      return Promise.resolve({
        data: { dispatch_mode: 'manual' },
        status: 200,
        statusText: 'OK',
        headers: {},
        config,
      });
    };

    await apiClient.get('/company_dispatch/mode');
    expect(capturedAuth).toBe('Bearer jwt-company');
  });
});
