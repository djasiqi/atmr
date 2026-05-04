import { describe, expect, it } from "@jest/globals";
import { computeAppViewport } from "./useAppViewport";
import { computePublicLanding } from "./useResponsiveTokens";

const zeroInsets = { top: 0, bottom: 0, left: 0, right: 0 };

describe("computePublicLanding", () => {
  const phone = computeAppViewport(400, 700, zeroInsets);

  it("isLargeText augmente le ratio lineHeight du titre (via titleLineHeight)", () => {
    const normal = computePublicLanding(phone, 1, false, false, 1);
    const large = computePublicLanding(phone, 1.2, true, false, 1.2);
    expect(large.titleLineHeight).toBeGreaterThanOrEqual(normal.titleLineHeight);
  });

  it("isVeryLargeText active stackSecondaryLinks même avec largeur confortable", () => {
    const wide = computePublicLanding(phone, 1, false, true, 1);
    expect(wide.stackSecondaryLinks).toBe(true);
  });

  it("minTouchHeight implicite : CTA height augmente avec fontScale", () => {
    const small = computePublicLanding(phone, 1, false, false, 1);
    const big = computePublicLanding(phone, 1.35, false, false, 1.25);
    expect(big.ctaHeight).toBeGreaterThanOrEqual(small.ctaHeight);
  });

  it("contentMaxWidth ne dépasse pas contentWidth du viewport", () => {
    const v = computeAppViewport(320, 640, zeroInsets);
    const landing = computePublicLanding(v, 1, false, false, 1);
    expect(landing.contentMaxWidth).toBeLessThanOrEqual(v.contentWidth);
  });

  it("cardPadding landing reste cohérent tiny vs tablet", () => {
    const tiny = computeAppViewport(340, 640, zeroInsets);
    const tablet = computeAppViewport(900, 1200, zeroInsets);
    const padTiny = computePublicLanding(tiny, 1, false, false, 1).cardPadding;
    const padTablet = computePublicLanding(tablet, 1, false, false, 1).cardPadding;
    expect(padTiny).toBeGreaterThan(0);
    expect(padTablet).toBeGreaterThanOrEqual(padTiny);
  });
});
