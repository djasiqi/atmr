/**
 * XSS InfoWindow — client_short (adresse) ne doit jamais être injecté brut.
 */
const {
  createStyledTooltip,
  escapeHtml,
} = require('../../pages/company/Dashboard/components/DriverLiveMap');

describe('DriverLiveMap tooltip XSS', () => {
  it('échappe une adresse malveillante dans metaLine', () => {
    const html = createStyledTooltip(
      { first_name: 'Jean', last_name: 'Dupont' },
      {
        status: 'busy',
        currentBookingId: 42,
        clientShort: '<img src=x onerror=alert(1)>',
      }
    );
    expect(html).not.toContain('<img src=x onerror=alert(1)>');
    expect(html).toContain('&lt;img src=x onerror=alert(1)&gt;');
    expect(html).toContain('Mission #42');
  });

  it('escapeHtml neutralise les balises', () => {
    expect(escapeHtml('<script>')).toBe('&lt;script&gt;');
  });
});
