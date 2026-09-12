import {
  validatePersistedLogoUrl,
  resolvePreviewBlobUrl,
} from '../logoUrlPolicy';
import { resolveLogoUrl } from '../resolveLogoUrl';

describe('politique URL de logo', () => {
  const jsAlert = `javascript${':'}alert(1)`;
  const jsAlertMixed = `JaVaScRiPt${':'}alert(1)`;

  it.each([
    jsAlert,
    jsAlertMixed,
    ` ${jsAlert}`,
    'data:text/html,<script>alert(1)</script>',
    'data:text/javascript,alert(1)',
    'data:image/png;base64,aaaa',
    'file:///etc/passwd',
    'vbscript:msgbox(1)',
    '//evil.example/logo.png',
    'http://evil.example/logo.png',
    'blob:https://example.com/123',
    '/etc/passwd',
    '/uploads/../secret.png',
    '/uploads//evil.png',
    'https://trusted.example@evil.example/logo.png',
  ])('refuse la persistance de %s', (value) => {
    expect(validatePersistedLogoUrl(value).ok).toBe(false);
    expect(resolveLogoUrl(value)).toBe('');
  });

  it('accepte /uploads et https', () => {
    expect(validatePersistedLogoUrl('/uploads/company_logos/logo.png')).toEqual({
      ok: true,
      value: '/uploads/company_logos/logo.png',
    });
    expect(validatePersistedLogoUrl('https://cdn.example.com/logo.webp')).toEqual({
      ok: true,
      value: 'https://cdn.example.com/logo.webp',
    });
    expect(resolveLogoUrl('https://cdn.example.com/logo.png')).toBe(
      'https://cdn.example.com/logo.png'
    );
  });

  it('autorise blob uniquement en preview', () => {
    const blobUrl = 'blob:http://localhost:3000/preview-1';
    expect(validatePersistedLogoUrl(blobUrl).ok).toBe(false);
    expect(resolveLogoUrl(blobUrl)).toBe('');
    expect(resolvePreviewBlobUrl(blobUrl)).toBe(blobUrl);
    expect(resolveLogoUrl(blobUrl, { allowPreview: true })).toBe(blobUrl);
  });

  it('vide une valeur invalide pour le fallback visuel', () => {
    expect(resolveLogoUrl(`javascript${':'}alert(1)`)).toBe('');
    expect(resolveLogoUrl('data:text/html,<h1>x</h1>')).toBe('');
  });
});
