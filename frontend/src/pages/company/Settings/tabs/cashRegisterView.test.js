import { allocationLabel, cashControlCsv, filterCashCases } from './cashRegisterView';

const rows = [
  {
    id: 1,
    driver_id: 4,
    booking_id: 12,
    created_at: '2026-10-27T09:00:00+00:00',
    tendered_chf: '50.00',
    change_chf: '5.00',
    net_chf: '45.00',
    state: 'received',
    handover_state: 'declared',
    reconciliation_variance_chf: null,
  },
  {
    id: 2,
    driver_id: 9,
    booking_id: 13,
    created_at: '2026-10-20T09:00:00+00:00',
    tendered_chf: '30.00',
    change_chf: '0.00',
    net_chf: '0.00',
    state: 'administrative',
    handover_state: null,
    reconciliation_variance_chf: null,
  },
];

describe('libellé d’imputation', () => {
  it('distingue non imputé, facture liée et information indisponible', () => {
    expect(allocationLabel({ allocation_status: 'not_allocated' })).toBe('Non imputé');
    expect(allocationLabel({
      allocation_status: 'allocated',
      invoice_number: 'EM-2026-11-0004',
      allocated_invoice_payment_id: 225,
    })).toBe('Facture EM-2026-11-0004');
    expect(allocationLabel({
      allocation_status: 'allocated',
      allocated_invoice_payment_id: 225,
    })).toBe('Imputé à une facture');
    expect(allocationLabel({ allocation_status: 'unavailable' })).toBe(
      'Information d’imputation indisponible',
    );
    expect(allocationLabel(null)).toBe('Information d’imputation indisponible');
  });
});

describe('filtres du registre de caisse', () => {
  it('isole un chauffeur, une remise et une exception', () => {
    expect(filterCashCases(rows, { driver: '4' })).toHaveLength(1);
    expect(filterCashCases(rows, { handover: 'declared' })[0].booking_id).toBe(12);
    expect(filterCashCases(rows, { status: 'administrative' })[0].booking_id).toBe(13);
    expect(filterCashCases(rows, { from: '2026-10-25' })).toHaveLength(1);
  });

  it('exporte un contrôle fictif sans présenter la conservation comme une recette', () => {
    const csv = cashControlCsv(rows);
    expect(csv).toContain('50.00;5.00;45.00;received;declared');
    expect(csv).toContain('administrative');
    expect(csv).not.toContain('recette définitive');
  });

  it('ajoute l’écart sans retirer les colonnes déjà exportées', () => {
    const csv = cashControlCsv([
      {
        ...rows[0],
        id: 8,
        reconciliation_variance_chf: '-5.00',
        reconciliations: [
          {
            handover_id: 3,
            expected_chf: '90.00',
            counted_chf: '85.00',
            variance_chf: '-5.00',
            reason: 'manque',
            status: 'constate',
          },
        ],
      },
    ]);
    expect(csv).toContain('50.00;5.00;45.00;received;declared');
    expect(csv).toContain(';8;3;90.00;85.00;-5.00;manque;constate');
  });
});
