import {
  isAnonymousPublicPath,
  isGuestSaferpayReturnPath,
  requestAuthNavigate,
  AUTH_NAVIGATE_EVENT,
} from './authNavigation';

describe('retour Saferpay invité', () => {
  it('reconnaît seulement la page publique de retour', () => {
    expect(isGuestSaferpayReturnPath('/guest/payment/saferpay/return')).toBe(true);
    expect(isGuestSaferpayReturnPath('/dashboard/company/1')).toBe(false);
    expect(isGuestSaferpayReturnPath('/dashboard/institution/1')).toBe(false);
    expect(isGuestSaferpayReturnPath('/driver/map')).toBe(false);
    expect(isGuestSaferpayReturnPath('/admin')).toBe(false);
  });

  it('ne redirige pas cette page vers la connexion', () => {
    const handler = jest.fn();
    window.addEventListener(AUTH_NAVIGATE_EVENT, handler);
    const original = window.location;
    delete window.location;
    window.location = { pathname: '/guest/payment/saferpay/return', replace: jest.fn(), assign: jest.fn() };
    requestAuthNavigate('/login?next=%2Fdashboard');
    expect(handler).not.toHaveBeenCalled();
    window.location = { pathname: '/dashboard/company/1', replace: jest.fn(), assign: jest.fn() };
    requestAuthNavigate('/login');
    expect(handler).toHaveBeenCalledTimes(1);
    window.removeEventListener(AUTH_NAVIGATE_EVENT, handler);
    window.location = original;
  });
});

describe('pages publiques AUTH-PRO anonymes', () => {
  it('autorise les parcours professionnels publics sans session', () => {
    expect(isAnonymousPublicPath('/professionnel')).toBe(true);
    expect(isAnonymousPublicPath('/inscription/organisation')).toBe(true);
    expect(isAnonymousPublicPath('/inscription/organisation/entreprise')).toBe(true);
    expect(isAnonymousPublicPath('/inscription/organisation/institution')).toBe(true);
    expect(isAnonymousPublicPath('/inscription/organisation/verifier')).toBe(true);
    expect(isAnonymousPublicPath('/inscription/organisation/suivi')).toBe(true);
    expect(isAnonymousPublicPath('/invitation/chauffeur')).toBe(true);
    expect(isAnonymousPublicPath('/invitation/chauffeur?token=abc')).toBe(true);
    expect(isAnonymousPublicPath('/')).toBe(true);
    expect(isAnonymousPublicPath('/conduire')).toBe(true);
  });

  it('refuse les portails protégés', () => {
    expect(isAnonymousPublicPath('/dashboard/company/1')).toBe(false);
    expect(isAnonymousPublicPath('/dashboard/institution/1')).toBe(false);
    expect(isAnonymousPublicPath('/driver/map')).toBe(false);
    expect(isAnonymousPublicPath('/admin')).toBe(false);
    expect(isAnonymousPublicPath('/book/new')).toBe(false);
  });

  it('ne redirige pas les pages AUTH-PRO vers /login sur session morte', () => {
    const handler = jest.fn();
    window.addEventListener(AUTH_NAVIGATE_EVENT, handler);
    const original = window.location;
    delete window.location;

    window.location = {
      pathname: '/inscription/organisation/entreprise',
      replace: jest.fn(),
      assign: jest.fn(),
    };
    requestAuthNavigate('/login?next=%2Finscription%2Forganisation%2Fentreprise');
    expect(handler).not.toHaveBeenCalled();

    window.location = {
      pathname: '/invitation/chauffeur',
      search: '?token=secret',
      replace: jest.fn(),
      assign: jest.fn(),
    };
    requestAuthNavigate('/login');
    expect(handler).not.toHaveBeenCalled();

    window.location = {
      pathname: '/professionnel',
      replace: jest.fn(),
      assign: jest.fn(),
    };
    requestAuthNavigate('/login');
    expect(handler).not.toHaveBeenCalled();

    window.location = {
      pathname: '/dashboard/company/xyz',
      replace: jest.fn(),
      assign: jest.fn(),
    };
    requestAuthNavigate('/login');
    expect(handler).toHaveBeenCalledTimes(1);

    window.removeEventListener(AUTH_NAVIGATE_EVENT, handler);
    window.location = original;
  });
});
