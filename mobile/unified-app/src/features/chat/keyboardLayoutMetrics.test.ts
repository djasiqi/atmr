import {
  computeEffectiveKeyboardTopY,
  computeFooterLiftCorrection,
  computeOemSafetyMargin,
  computeVisibleBottomInsets,
  shellFooterOffset,
} from "./keyboardLayoutMetrics";

describe("computeVisibleBottomInsets", () => {
  it("priorise resizeDelta quand adjustResize est actif", () => {
    const result = computeVisibleBottomInsets({
      baselineWindowHeight: 780,
      windowHeight: 453,
      keyboardHeight: 300,
      screenY: 453,
    });
    expect(result.resizeDelta).toBe(327);
    expect(result.visibleBottomInset).toBe(327);
  });

  it("petit slack — toolbar partielle au-dessus de screenY (Samsung)", () => {
    const result = computeVisibleBottomInsets({
      baselineWindowHeight: 780,
      windowHeight: 780,
      keyboardHeight: 327,
      screenY: 438,
    });
    expect(result.measuredSlack).toBe(15);
    expect(result.effectiveKeyboardTopY).toBe(423);
    expect(result.visibleBottomInset).toBe(357);
  });

  it("grand slack — screenY au niveau touches (Gboard Pixel)", () => {
    const result = computeVisibleBottomInsets({
      baselineWindowHeight: 997,
      windowHeight: 997,
      keyboardHeight: 327,
      screenY: 622,
    });
    expect(result.measuredSlack).toBe(48);
    expect(result.effectiveKeyboardTopY).toBe(574);
    expect(result.visibleBottomInset).toBe(423);
  });

  it("guard Samsung screenY=0 : infere via keyboardHeight", () => {
    const result = computeVisibleBottomInsets({
      baselineWindowHeight: 780,
      windowHeight: 780,
      keyboardHeight: 327,
      screenY: 0,
    });
    expect(result.measuredSlack).toBe(0);
    expect(result.effectiveKeyboardTopY).toBe(453);
    expect(result.visibleBottomInset).toBe(327);
  });

  it("guard Samsung screenY=windowHeight : infere via keyboardHeight", () => {
    const result = computeVisibleBottomInsets({
      baselineWindowHeight: 780,
      windowHeight: 780,
      keyboardHeight: 327,
      screenY: 780,
    });
    expect(result.measuredSlack).toBe(0);
    expect(result.effectiveKeyboardTopY).toBe(453);
    expect(result.visibleBottomInset).toBe(327);
  });

  it("computeEffectiveKeyboardTopY", () => {
    expect(computeEffectiveKeyboardTopY(622, 48)).toBe(574);
    expect(computeEffectiveKeyboardTopY(438, 15)).toBe(423);
    expect(computeEffectiveKeyboardTopY(438, 0)).toBe(438);
  });

  it("shellFooterOffset convertit inset fenêtre en offset shell", () => {
    expect(shellFooterOffset(423, 178)).toBe(245);
    expect(shellFooterOffset(357, 178)).toBe(179);
  });

  it("liftCorrection corrige uniquement les chevauchements positifs", () => {
    expect(computeFooterLiftCorrection(430, 423)).toBe(7);
    expect(computeFooterLiftCorrection(424, 423)).toBe(0);
  });

  it("liftCorrection respecte la marge de sécurité OEM (footer au-dessus)", () => {
    expect(computeFooterLiftCorrection(415, 423)).toBe(0);
    expect(computeFooterLiftCorrection(400, 423)).toBe(0);
  });

  it("computeOemSafetyMargin = 8 px par défaut (fontScale ≤ 1)", () => {
    expect(computeOemSafetyMargin(0.8)).toBe(8);
    expect(computeOemSafetyMargin(1)).toBe(8);
  });

  it("computeOemSafetyMargin scale au-dessus de 1.0", () => {
    expect(computeOemSafetyMargin(1.15)).toBe(12);
    expect(computeOemSafetyMargin(1.3)).toBe(15);
  });

  it("computeOemSafetyMargin est plafonnée à 20 px", () => {
    expect(computeOemSafetyMargin(2)).toBe(20);
    expect(computeOemSafetyMargin(3)).toBe(20);
  });

  it("computeOemSafetyMargin résiste aux valeurs invalides", () => {
    expect(computeOemSafetyMargin(Number.NaN)).toBe(8);
    expect(computeOemSafetyMargin(Number.POSITIVE_INFINITY)).toBe(8);
  });
});
