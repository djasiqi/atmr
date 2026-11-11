import { registryStatusLabel } from './invoiceRegistryStatus';

describe('libellé de statut du registre', () => {
  test('un avoir qui solde sans paiement affiche Solde nul', () => {
    expect(
      registryStatusLabel({ status: 'paid', amount_paid: 0, balance_due: 0 })
    ).toBe('Solde nul');
  });

  test('un règlement qui solde affiche Payée', () => {
    expect(
      registryStatusLabel({ status: 'paid', amount_paid: 129.25, balance_due: 0 })
    ).toBe('Payée');
  });

  test('un règlement partiel reste Partiellement payée', () => {
    expect(
      registryStatusLabel({
        status: 'partially_paid',
        amount_paid: 60,
        balance_due: 69.25,
      })
    ).toBe('Partiellement payée');
  });
});
