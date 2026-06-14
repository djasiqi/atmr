import { getCarrierDisplay } from '../carrierDisplay';

describe('getCarrierDisplay', () => {
  it('lit le snapshot transporteur externe', () => {
    const view = getCarrierDisplay({
      carrier_source: 'external',
      external_carrier: {
        name: 'Taxi Urgence',
        phone: '+41 79 000 00 00',
        reference: 'REF-1',
        reason: 'Dépannage',
      },
    });
    expect(view.type).toBe('external');
    expect(view.name).toBe('Taxi Urgence');
    expect(view.badge).toBe('EXTERNE');
    expect(view.reason).toBe('Dépannage');
  });

  it('lit le transporteur LIRIE', () => {
    const view = getCarrierDisplay({
      carrier_source: 'lirie',
      accepted_by_company: {
        name: 'EM Transport',
        contact_phone: '+41 22 000 00 00',
        contact_email: 'ops@em.ch',
      },
    });
    expect(view.type).toBe('lirie');
    expect(view.name).toBe('EM Transport');
    expect(view.email).toBe('ops@em.ch');
    expect(view.badge).toBeNull();
  });
});
