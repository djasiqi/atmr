import { buildSafeAppPath, pathFromCompanyReturnTo, pathFromNextQueryParam } from './safeReturnPath';

describe('safeReturnPath', () => {
  beforeEach(() => {
    Object.defineProperty(window, 'location', {
      configurable: true,
      writable: true,
      value: {
        origin: 'https://app.example.com',
        href: 'https://app.example.com/',
      },
    });
  });

  describe('buildSafeAppPath', () => {
    it('accepte un chemin interne avec query', () => {
      expect(buildSafeAppPath('/client/payment/worldline/return', '?bookingId=3')).toBe(
        '/client/payment/worldline/return?bookingId=3'
      );
    });

    it('refuse les open redirects', () => {
      expect(buildSafeAppPath('//evil.com')).toBeNull();
      expect(buildSafeAppPath('https://evil.com')).toBeNull();
      expect(buildSafeAppPath('/../admin')).toBeNull();
    });

    it('refuse /login', () => {
      expect(buildSafeAppPath('/login')).toBeNull();
    });
  });

  describe('pathFromNextQueryParam', () => {
    it('décode un next interne', () => {
      const encoded = encodeURIComponent('/client/payment/worldline/return?bookingId=9');
      expect(pathFromNextQueryParam(encoded)).toBe('/client/payment/worldline/return?bookingId=9');
    });

    it('refuse une autre origine', () => {
      expect(pathFromNextQueryParam(encodeURIComponent('https://evil.com/x'))).toBeNull();
    });
  });

  describe('pathFromCompanyReturnTo', () => {
    it('accepte un chemin entreprise exact', () => {
      expect(pathFromCompanyReturnTo('/dashboard/company/1/clients')).toBe(
        '/dashboard/company/1/clients'
      );
      expect(pathFromCompanyReturnTo('/company/invoices?tab=1')).toBe('/company/invoices?tab=1');
    });

    it('refuse les lookalikes, userinfo et protocoles dangereux', () => {
      expect(pathFromCompanyReturnTo('https://www.lirie.ch.evil.example/')).toBeNull();
      expect(pathFromCompanyReturnTo('https://www.lirie.ch@evil.example/')).toBeNull();
      expect(pathFromCompanyReturnTo('https://evil.example/www.lirie.ch')).toBeNull();
      // eslint-disable-next-line no-script-url -- cas adversarial de protocole, jamais exécuté
      expect(pathFromCompanyReturnTo('javascript:alert(1)')).toBeNull();
      expect(pathFromCompanyReturnTo('data:text/html,hi')).toBeNull();
      expect(pathFromCompanyReturnTo('file:///etc/passwd')).toBeNull();
      expect(pathFromCompanyReturnTo('//evil.example/company/x')).toBeNull();
      expect(pathFromCompanyReturnTo('https://app.example.com:444/company/x')).toBeNull();
    });

    it('refuse un préfixe hors entreprise et les traversals', () => {
      expect(pathFromCompanyReturnTo('/dashboard/institution/1')).toBeNull();
      expect(pathFromCompanyReturnTo('/company')).toBeNull();
      expect(pathFromCompanyReturnTo('/dashboard/company/../admin')).toBeNull();
    });
  });
});
