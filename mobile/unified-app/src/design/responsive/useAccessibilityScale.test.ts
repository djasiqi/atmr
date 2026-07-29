import { computeAccessibilityScale } from "./useAccessibilityScale";
import { CHROME_FONT_CAP, CONTENT_FONT_CAP } from "./fontScaleCaps";

describe("computeAccessibilityScale", () => {
  it("expose les caps contenu / chrome", () => {
    const s = computeAccessibilityScale(1, 390);
    expect(s.contentMaxFontMultiplier).toBe(CONTENT_FONT_CAP);
    expect(s.chromeMaxFontMultiplier).toBe(CHROME_FONT_CAP);
  });

  it("isLargeText à partir de 1.15", () => {
    expect(computeAccessibilityScale(1.14, 390).isLargeText).toBe(false);
    expect(computeAccessibilityScale(1.15, 390).isLargeText).toBe(true);
  });

  it("isVeryLargeText à partir de 1.3", () => {
    expect(computeAccessibilityScale(1.29, 390).isVeryLargeText).toBe(false);
    expect(computeAccessibilityScale(1.3, 390).isVeryLargeText).toBe(true);
  });

  it("shouldStackRows si usableWidth === 360 même à police normale", () => {
    expect(computeAccessibilityScale(1, 360).shouldStackRows).toBe(true);
  });

  it("shouldStackRows si usableWidth < 360", () => {
    expect(computeAccessibilityScale(1, 320).shouldStackRows).toBe(true);
  });

  it("ne stack pas à police normale si usableWidth > 360", () => {
    expect(computeAccessibilityScale(1, 361).shouldStackRows).toBe(false);
    expect(computeAccessibilityScale(1.2, 390).shouldStackRows).toBe(false);
  });

  it("shouldStackRows si isVeryLargeText même sur large écran", () => {
    expect(computeAccessibilityScale(1.3, 430).shouldStackRows).toBe(true);
  });
});
