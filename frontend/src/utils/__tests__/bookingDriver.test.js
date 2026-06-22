import { describe, expect, it } from '@jest/globals';
import { resolveBookingDriverName } from '../bookingDriver';

describe('resolveBookingDriverName', () => {
  it('privilégie driver.full_name', () => {
    expect(
      resolveBookingDriverName({
        driver_name: 'None None',
        driver: { full_name: 'Emmenez Moi' },
      }),
    ).toBe('Emmenez Moi');
  });

  it('ignore driver_name invalide « None None »', () => {
    expect(
      resolveBookingDriverName({
        driver_name: 'None None',
        driver: { username: 'Emmenez Moi' },
      }),
    ).toBe('Emmenez Moi');
  });

  it('retombe sur driver_id', () => {
    expect(resolveBookingDriverName({ driver_id: 42 })).toBe('Chauffeur #42');
  });
});
