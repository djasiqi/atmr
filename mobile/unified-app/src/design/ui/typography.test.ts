import { describe, expect, it } from "@jest/globals";
import { computeAppViewport } from "../responsive/useAppViewport";
import { computePublicLanding } from "../responsive/useResponsiveTokens";
import { getAppTextStyle } from "./typography";

const zeroInsets = { top: 0, bottom: 0, left: 0, right: 0 };

describe("getAppTextStyle", () => {
  const viewport = computeAppViewport(400, 800, zeroInsets);
  const tokensStub = {
    fontScale: 1,
    effectiveFontScale: 1,
    scrollExtraBottomPadding: 0,
    minTouchHeight: 44,
    fieldShellMinHeight: 40,
    fieldTextInputMinHeight: 38,
    fieldTextInputPaddingV: 10,
    formButtonMinHeight: 44,
    spacingXs: 4,
    spacingSm: 8,
    spacingMd: 16,
    spacingLg: 24,
    radiusSm: 8,
    radiusMd: 12,
    bodyLineHeightRatio: 1.25,
    headingLineHeightRatio: 1.22,
    pageGap: 24,
    sectionGap: 16,
    fieldGap: 8,
    cardPadding: 18,
    bodyFontSize: 16,
    buttonFontSize: 13,
    radiusLg: 16,
    landing: computePublicLanding(viewport, 1, false, false, 1),
  };

  it("screenTitle est plus grand que body", () => {
    const a = getAppTextStyle("screenTitle", tokensStub as any, viewport);
    const b = getAppTextStyle("body", tokensStub as any, viewport);
    expect((a.fontSize as number) ?? 0).toBeGreaterThan((b.fontSize as number) ?? 0);
  });

  it("error utilise une couleur distincte du corps", () => {
    const e = getAppTextStyle("error", tokensStub as any, viewport);
    const body = getAppTextStyle("body", tokensStub as any, viewport);
    expect(e.color).not.toBe(body.color);
  });
});
