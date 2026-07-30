import { describe, expect, it, jest } from "@jest/globals";
import { createElement } from "react";
import TestRenderer, { act } from "react-test-renderer";
import { Text } from "react-native";
import { CONTENT_FONT_CAP, CHROME_FONT_CAP } from "./fontScaleCaps";

jest.mock("./useResponsiveTokens", () => ({
  useResponsiveTokens: () => ({
    fontScale: 1,
    effectiveFontScale: 1,
    verticalLayoutScale: 1,
    densityScale: 1,
    radiusScale: 1,
    bodyFontSize: 16,
    bodyLineHeightRatio: 1.25,
    headingLineHeightRatio: 1.22,
    spacingSm: 8,
    spacingMd: 16,
    spacingLg: 24,
    spacingXs: 4,
    radiusSm: 8,
    radiusMd: 12,
    radiusLg: 16,
    pageGap: 24,
    sectionGap: 16,
    fieldGap: 8,
    cardPadding: 16,
    buttonFontSize: 13,
    formButtonMinHeight: 44,
    fieldShellMinHeight: 44,
    fieldTextInputMinHeight: 42,
    fieldTextInputPaddingV: 10,
    minTouchHeight: 44,
    scrollExtraBottomPadding: 20,
    keyboardScrollPaddingMin: 260,
    keyboardScrollPaddingExtra: 48,
    dropdownListMaxHeight: 280,
    modalSheetMaxHeightRatio: 0.88,
    modalSheetMaxHeightCap: 620,
    landing: {},
  }),
}));

jest.mock("./useAppViewport", () => ({
  useAppViewport: () => ({
    width: 390,
    height: 800,
    usableWidth: 390,
    usableHeight: 760,
    contentWidth: 354,
    horizontalPadding: 18,
    isTiny: false,
    isCompact: true,
    isRegular: false,
    isTablet: false,
  }),
}));

jest.mock("./useAccessibilityScale", () => ({
  useAccessibilityScale: () => ({
    fontScale: 1,
    isLargeText: false,
    isVeryLargeText: false,
    shouldStackRows: false,
    contentMaxFontMultiplier: 2,
    chromeMaxFontMultiplier: 1.3,
  }),
}));

const { AppText } = require("../ui/AppText");

describe("AppText scaleRole", () => {
  it("applique CONTENT_FONT_CAP par défaut (content)", () => {
    let tree: TestRenderer.ReactTestRenderer;
    act(() => {
      tree = TestRenderer.create(createElement(AppText, { variant: "body" }, "Hello"));
    });
    const text = tree!.root.findByType(Text);
    expect(text.props.maxFontSizeMultiplier).toBe(CONTENT_FONT_CAP);
  });

  it("applique CHROME_FONT_CAP pour scaleRole=chrome", () => {
    let tree: TestRenderer.ReactTestRenderer;
    act(() => {
      tree = TestRenderer.create(
        createElement(AppText, { variant: "caption", scaleRole: "chrome" }, "Tab")
      );
    });
    const text = tree!.root.findByType(Text);
    expect(text.props.maxFontSizeMultiplier).toBe(CHROME_FONT_CAP);
  });

  it("laisse maxFontSizeMultiplier explicite primer", () => {
    let tree: TestRenderer.ReactTestRenderer;
    act(() => {
      tree = TestRenderer.create(
        createElement(AppText, { variant: "body", maxFontSizeMultiplier: 1.1 }, "X")
      );
    });
    const text = tree!.root.findByType(Text);
    expect(text.props.maxFontSizeMultiplier).toBe(1.1);
  });
});
