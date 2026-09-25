import { requiresPrivateOnlinePaymentAtBooking } from './clientBookingPayment';

describe('requiresPrivateOnlinePaymentAtBooking', () => {
  it('retourne true pour billed_to_type patient', () => {
    expect(requiresPrivateOnlinePaymentAtBooking({ billed_to_type: 'patient' })).toBe(true);
    expect(
      requiresPrivateOnlinePaymentAtBooking({ billing: { billed_to_type: 'patient' } })
    ).toBe(true);
  });

  it('retourne false pour clinique ou assurance', () => {
    expect(requiresPrivateOnlinePaymentAtBooking({ billed_to_type: 'clinic' })).toBe(false);
    expect(requiresPrivateOnlinePaymentAtBooking({ billed_to_type: 'insurance' })).toBe(false);
    expect(
      requiresPrivateOnlinePaymentAtBooking({ billing: { billed_to_type: 'insurance' } })
    ).toBe(false);
  });

  it('retourne false pour un client PORTAL (pas de Saferpay)', () => {
    expect(
      requiresPrivateOnlinePaymentAtBooking({
        billed_to_type: 'patient',
        client: { client_type: 'PORTAL' },
      })
    ).toBe(false);
    expect(
      requiresPrivateOnlinePaymentAtBooking({
        billed_to_type: 'patient',
        portal_contract_flow: 'double_validation_v2',
      })
    ).toBe(false);
  });

  it('par défaut (données absentes) considère le paiement client', () => {
    expect(requiresPrivateOnlinePaymentAtBooking(null)).toBe(true);
    expect(requiresPrivateOnlinePaymentAtBooking(undefined)).toBe(true);
  });
});
