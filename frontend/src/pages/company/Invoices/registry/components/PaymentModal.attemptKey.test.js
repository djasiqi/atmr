import { buildManualPaymentPayload } from './PaymentModal';

describe('paiement manuel — clé de rejeu', () => {
  test('la même tentative conserve sa clé même sans référence saisie', () => {
    const form = {
      amount: '',
      paid_at: '2026-11-11',
      method: 'bank_transfer',
      reference: '',
    };
    const first = buildManualPaymentPayload(form, 1, 'tentative-stable');
    const retry = buildManualPaymentPayload(form, 1, 'tentative-stable');
    expect(first.idempotency_key).toBe('tentative-stable');
    expect(retry.idempotency_key).toBe(first.idempotency_key);
    expect(first.reference).toBe('');
    expect(first.amount).toBe(1);
  });
});
