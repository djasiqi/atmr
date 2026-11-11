import {
  applyLocalLinePatch,
  applyLocalPercentDiscount,
  buildCreditLinesFromPreparation,
  cloneInvoiceForCorrection,
  money,
  removeLocalLine,
} from './correctionPreparation';

describe('correctionPreparation', () => {
  const original = {
    id: 3171,
    total_amount: 96.9,
    amount_paid: 20,
    lines: [
      {
        id: 5773,
        description: 'A',
        line_total: 45,
        vat_rate: 7.7,
        vat_amount: 3.45,
        total_with_vat: 48.45,
      },
      {
        id: 5774,
        description: 'B',
        line_total: 45,
        vat_rate: 7.7,
        vat_amount: 3.45,
        total_with_vat: 48.45,
      },
    ],
  };

  const residuals = [
    {
      invoice_line_id: 5773,
      description: 'A',
      amount_ttc: 48.45,
      credited_ttc: 24.23,
      residual_ttc: 24.22,
    },
    {
      invoice_line_id: 5774,
      description: 'B',
      amount_ttc: 48.45,
      credited_ttc: 0,
      residual_ttc: 48.45,
    },
  ];

  it('clone puis réduction de montant construit un avoir partiel', () => {
    let prep = cloneInvoiceForCorrection(original);
    // Ligne B sans crédit préalable : 45 HT → 22.5 HT ≈ 24.23 TTC
    prep = applyLocalLinePatch(prep, 5774, { line_total: 22.5 });
    const { creditLines, unsupported } = buildCreditLinesFromPreparation({
      originalInvoice: original,
      preparationInvoice: prep,
      residuals,
    });
    expect(unsupported).toHaveLength(0);
    expect(creditLines[0].invoice_line_id).toBe(5774);
    expect(money(creditLines[0].amount_ttc)).toBeGreaterThan(20);
  });

  it('retrait de ligne → crédit résiduel complet', () => {
    let prep = cloneInvoiceForCorrection(original);
    prep = removeLocalLine(prep, 5774);
    const { creditLines } = buildCreditLinesFromPreparation({
      originalInvoice: original,
      preparationInvoice: prep,
      residuals,
    });
    expect(creditLines).toContainEqual({ invoice_line_id: 5774 });
  });

  it('libellé seul sans baisse → unsupported', () => {
    let prep = cloneInvoiceForCorrection(original);
    prep = applyLocalLinePatch(prep, 5774, { description: 'B corrigé' });
    const { unsupported, creditLines } = buildCreditLinesFromPreparation({
      originalInvoice: original,
      preparationInvoice: prep,
      residuals,
    });
    expect(creditLines).toHaveLength(0);
    expect(unsupported[0].invoice_line_id).toBe(5774);
  });

  it('remise % locale réduit les HT et produit un crédit', () => {
    const prep0 = cloneInvoiceForCorrection(original);
    const { invoice: prep, error, touched } = applyLocalPercentDiscount(prep0, 10);
    expect(error).toBeNull();
    expect(touched).toBe(2);
    expect(money(prep.lines[0].line_total)).toBe(40.5);
    const { creditLines } = buildCreditLinesFromPreparation({
      originalInvoice: original,
      preparationInvoice: prep,
      residuals,
    });
    expect(creditLines.length).toBeGreaterThanOrEqual(1);
  });
});

