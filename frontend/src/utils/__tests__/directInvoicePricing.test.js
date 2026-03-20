import {
  roundTo005,
  isSameMoney,
  getDisplayedLineAmount,
  resolveDiscountBaseForLine,
  applyPercentDiscount,
} from '../directInvoicePricing';

describe('directInvoicePricing', () => {
  describe('roundTo005', () => {
    it('aligne sur les exemples facturation (0,05 CHF)', () => {
      expect(roundTo005(10.32)).toBe(10.3);
      expect(roundTo005(10.33)).toBe(10.35);
      expect(roundTo005(10.36)).toBe(10.35);
      expect(roundTo005(10.38)).toBe(10.4);
    });
  });

  describe('isSameMoney', () => {
    it('tolère les flottants dans la limite', () => {
      expect(isSameMoney(45, 45.004, 0.01)).toBe(true);
      expect(isSameMoney(45, 45.02, 0.01)).toBe(false);
    });
  });

  describe('getDisplayedLineAmount', () => {
    it('priorité override > amount > estimated', () => {
      const r = { amount: 0.5, estimated_amount: 12 };
      expect(getDisplayedLineAmount(r, {})).toBe(0.5);
      expect(getDisplayedLineAmount(r, { amount: 45 })).toBe(45);
      expect(getDisplayedLineAmount({ estimated_amount: 12 }, {})).toBe(12);
      expect(getDisplayedLineAmount({}, {})).toBe(0);
    });
  });

  describe('scénario bug historique + réapplication remise', () => {
    it('correction manuelle 45 puis 25% → 33,75 (pas le catalogue 0,50)', () => {
      const reservation = { id: 1, amount: 0.5, estimated_amount: 0.5 };
      const prevOverride = { amount: 45 };
      const pct = 25;
      const { chosenBaseAmount } = resolveDiscountBaseForLine({
        reservation,
        prevOverride,
        discountBaseMode: 'adjusted',
      });
      expect(chosenBaseAmount).toBe(45);
      expect(applyPercentDiscount(chosenBaseAmount, pct)).toBe(33.75);
    });

    it('double clic 25% ne compose pas : base stable via pricingMeta', () => {
      const reservation = { id: 1, amount: 0.5 };
      const pct = 25;
      const prevOverride = {
        amount: 33.75,
        pricingMeta: {
          baseBeforeDiscount: 45,
          discountPercent: 25,
          discountBaseMode: 'adjusted',
        },
      };
      const { chosenBaseAmount } = resolveDiscountBaseForLine({
        reservation,
        prevOverride,
        discountBaseMode: 'adjusted',
      });
      expect(chosenBaseAmount).toBe(45);
      expect(applyPercentDiscount(chosenBaseAmount, pct)).toBe(33.75);
    });
  });

  describe('resolveDiscountBaseForLine mode catalog', () => {
    it('ignore prev.amount et pricingMeta', () => {
      const reservation = { id: 1, amount: 10, estimated_amount: 99 };
      const prevOverride = {
        amount: 77,
        pricingMeta: { baseBeforeDiscount: 50, discountPercent: 10, discountBaseMode: 'adjusted' },
      };
      const { chosenBaseAmount } = resolveDiscountBaseForLine({
        reservation,
        prevOverride,
        discountBaseMode: 'catalog',
      });
      expect(chosenBaseAmount).toBe(10);
    });
  });
});
