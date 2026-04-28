import { buildSafeAppPath, pathFromNextQueryParam } from './safeReturnPath';

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
});
