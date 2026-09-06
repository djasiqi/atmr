import {
  hasCompanyDispatchSession,
  hasCompanyScopedAccessToken,
} from '../webAuthSession';

describe('hasCompanyDispatchSession', () => {
  beforeEach(() => {
    localStorage.clear();
    localStorage.setItem('lirie_auth_env', 'app');
  });

  it('est faux sans user ni token', () => {
    expect(hasCompanyScopedAccessToken()).toBe(false);
    expect(hasCompanyDispatchSession()).toBe(false);
  });

  it('accepte une session entreprise cookie-only (user, pas de JWT JS)', () => {
    localStorage.setItem(
      'app_user',
      JSON.stringify({ role: 'company', public_id: 'c-1' })
    );
    expect(hasCompanyScopedAccessToken()).toBe(false);
    expect(hasCompanyDispatchSession()).toBe(true);
  });

  it('accepte un admin cookie-only', () => {
    localStorage.setItem(
      'app_user',
      JSON.stringify({ role: 'admin', public_id: 'a-1' })
    );
    expect(hasCompanyDispatchSession()).toBe(true);
  });

  it('refuse un chauffeur', () => {
    localStorage.setItem(
      'app_user',
      JSON.stringify({ role: 'driver', public_id: 'd-1' })
    );
    expect(hasCompanyDispatchSession()).toBe(false);
  });

  it('accepte un JWT entreprise JS', () => {
    localStorage.setItem('company_access_token', 'jwt-company');
    expect(hasCompanyDispatchSession()).toBe(true);
  });
});
