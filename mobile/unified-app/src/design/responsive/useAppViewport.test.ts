import { describe, expect, it } from "@jest/globals";
import { computeAppViewport } from "./useAppViewport";

const zeroInsets = { top: 0, bottom: 0, left: 0, right: 0 };

describe("computeAppViewport", () => {
  it("safeTop / safeBottom sont au minimum 16", () => {
    const v = computeAppViewport(400, 800, zeroInsets);
    expect(v.safeTop).toBe(16);
    expect(v.safeBottom).toBe(16);
    expect(v.topInset).toBe(v.safeTop);
    expect(v.bottomInset).toBe(v.safeBottom);
  });

  it("respecte les insets natifs quand ils dépassent 16", () => {
    const v = computeAppViewport(400, 800, { top: 47, bottom: 34, left: 0, right: 0 });
    expect(v.safeTop).toBe(47);
    expect(v.safeBottom).toBe(34);
  });

  it("seuils shortest : tiny / compact / regular / tablet", () => {
    expect(computeAppViewport(359, 640, zeroInsets).isTiny).toBe(true);
    expect(computeAppViewport(359, 640, zeroInsets).isCompact).toBe(false);

    const c360 = computeAppViewport(360, 700, zeroInsets);
    expect(c360.isTiny).toBe(false);
    expect(c360.isCompact).toBe(true);
    expect(c360.isRegular).toBe(false);

    const c399 = computeAppViewport(399, 700, zeroInsets);
    expect(c399.isCompact).toBe(true);

    const r400 = computeAppViewport(400, 800, zeroInsets);
    expect(r400.isRegular).toBe(true);
    expect(r400.isTablet).toBe(false);

    const tab = computeAppViewport(800, 1024, zeroInsets);
    expect(tab.isTablet).toBe(true);
    expect(tab.shortest).toBe(800);

    const edge767 = computeAppViewport(767, 1024, zeroInsets);
    expect(edge767.isRegular).toBe(true);
    expect(edge767.isTablet).toBe(false);

    const edge768 = computeAppViewport(768, 1024, zeroInsets);
    expect(edge768.isTablet).toBe(true);
  });

  it("contentWidth dérive de usableWidth et gutters, sans largeur min 288 artificielle", () => {
    const narrow = computeAppViewport(280, 600, zeroInsets);
    expect(narrow.usableWidth).toBe(280);
    expect(narrow.horizontalPadding).toBe(14);
    expect(narrow.contentWidth).toBe(Math.min(280 - 2 * 14, 440));

    const wide = computeAppViewport(1200, 800, zeroInsets);
    expect(wide.isLandscape).toBe(true);
    expect(wide.usableWidth).toBe(1200);
    expect(wide.contentWidth).toBe(520);
  });

  it("paysage : usableWidth soustrait les encarts latéraux", () => {
    const v = computeAppViewport(900, 400, { top: 0, bottom: 0, left: 24, right: 24 });
    expect(v.isLandscape).toBe(true);
    expect(v.usableWidth).toBe(900 - 48);
    expect(v.contentWidth).toBeLessThanOrEqual(440);
  });
});
