import {
  roundTo005,
  isSameMoney,
  getDisplayedLineAmount,
  computeGlobalPercentDiscountOnSubtotal,
  parseGlobalDiscountPercentField,
  lineOverrideMaySuggestPriorDiscount,
  directSelectionMaySuggestLineLevelDiscount,
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

  describe('parseGlobalDiscountPercentField', () => {
    it('accepte 25, 25 %, 12,5 et refuse brouillon / hors plage', () => {
      expect(parseGlobalDiscountPercentField('25')).toBe(25);
      expect(parseGlobalDiscountPercentField('12,5')).toBe(12.5);
      expect(parseGlobalDiscountPercentField('')).toBe(null);
      expect(parseGlobalDiscountPercentField('25.')).toBe(null);
      expect(parseGlobalDiscountPercentField('0')).toBe(null);
      expect(parseGlobalDiscountPercentField('101')).toBe(null);
      expect(parseGlobalDiscountPercentField('abc')).toBe(null);
    });
  });

  describe('computeGlobalPercentDiscountOnSubtotal', () => {
    it('12 × 90 CHF, remise 25 % → sous-total 1080, remise 270, net 810', () => {
      const sub = roundTo005(12 * 90);
      expect(sub).toBe(1080);
      const { discountAmountHt, netHtAfter } = computeGlobalPercentDiscountOnSubtotal(sub, 25);
      expect(discountAmountHt).toBe(270);
      expect(netHtAfter).toBe(810);
    });

    it('sous-total 1093.00 CHF, remise 25 % → remise 273.25 (arrondi unique sur le montant de remise)', () => {
      const sub = 1093;
      const { discountAmountHt, netHtAfter } = computeGlobalPercentDiscountOnSubtotal(sub, 25);
      expect(discountAmountHt).toBe(273.25);
      expect(netHtAfter).toBe(roundTo005(sub - discountAmountHt));
      expect(netHtAfter).toBe(819.75);
    });

    it('correction manuelle : ligne 45 CHF reste 45 ; remise globale sur sous-total', () => {
      const r = { id: 1, amount: 90, estimated_amount: 90 };
      const lineHt = getDisplayedLineAmount(r, { amount: 45 });
      expect(lineHt).toBe(45);
      const { discountAmountHt, netHtAfter } = computeGlobalPercentDiscountOnSubtotal(lineHt, 25);
      expect(discountAmountHt).toBe(roundTo005(45 * 0.25));
      expect(netHtAfter).toBe(roundTo005(45 - discountAmountHt));
    });

    it('panier mixte : 10×90 + 1×45 → sous-total 945, remise 25 % sur le tout', () => {
      const sub = roundTo005(10 * 90 + 45);
      expect(sub).toBe(945);
      const { discountAmountHt, netHtAfter } = computeGlobalPercentDiscountOnSubtotal(sub, 25);
      expect(discountAmountHt).toBe(236.25);
      expect(netHtAfter).toBe(708.75);
    });
  });

  describe('anti double remise (heuristique)', () => {
    it('détecte une note contenant « remise »', () => {
      expect(lineOverrideMaySuggestPriorDiscount({ note: 'Remise commerciale exceptionnelle' })).toBe(
        true
      );
      expect(lineOverrideMaySuggestPriorDiscount({ note: 'Transport standard' })).toBe(false);
    });

    it('détecte pricingMeta legacy', () => {
      expect(
        lineOverrideMaySuggestPriorDiscount({
          amount: 67.5,
          pricingMeta: { baseBeforeDiscount: 90, discountPercent: 25 },
        })
      ).toBe(true);
    });

    it('directSelectionMaySuggestLineLevelDiscount parcourt les overrides', () => {
      const res = [{ id: 1 }, { id: 2 }];
      const ov = { 1: { note: 'ok' }, 2: { note: 'remise 10%' } };
      expect(directSelectionMaySuggestLineLevelDiscount(res, ov)).toBe(true);
      expect(directSelectionMaySuggestLineLevelDiscount(res, { 1: {}, 2: {} })).toBe(false);
    });
  });

  describe('resolveDiscountBaseForLine (helpers hérités)', () => {
    it('mode adjusted : base stable via pricingMeta.baseBeforeDiscount', () => {
      const reservation = { id: 1, amount: 0.5 };
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
      expect(applyPercentDiscount(chosenBaseAmount, 25)).toBe(33.75);
    });

    it('mode catalog : ignore prev.amount et pricingMeta', () => {
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
