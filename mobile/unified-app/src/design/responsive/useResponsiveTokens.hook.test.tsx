import { beforeEach, describe, expect, it, jest } from "@jest/globals";
import { createElement, useEffect } from "react";
import TestRenderer, { act } from "react-test-renderer";
import type { AppViewport } from "./useAppViewport";
import { useResponsiveTokens } from "./useResponsiveTokens";

const zeroInsets = { top: 0, bottom: 0, left: 0, right: 0 };

const mockUseAppViewport = jest.fn<() => AppViewport>();
const mockUseAccessibilityScale = jest.fn();

jest.mock("./useAppViewport", () => ({
  useAppViewport: () => mockUseAppViewport(),
}));

jest.mock("./useAccessibilityScale", () => ({
  useAccessibilityScale: () => mockUseAccessibilityScale(),
}));

function HookCapture(props: { onValue: (v: ReturnType<typeof useResponsiveTokens>) => void }) {
  const v = useResponsiveTokens();
  useEffect(() => {
    props.onValue(v);
  });
  return null;
}

describe("useResponsiveTokens (hook)", () => {
  beforeEach(() => {
    const { computeAppViewport } = jest.requireActual<typeof import("./useAppViewport")>("./useAppViewport");
    mockUseAppViewport.mockReturnValue(computeAppViewport(400, 800, zeroInsets));
  });

  it("minTouchHeight augmente quand isLargeText et fontScale > 1", () => {
    let captured: ReturnType<typeof useResponsiveTokens> | undefined;
    mockUseAccessibilityScale.mockReturnValue({
      fontScale: 1.2,
      isLargeText: true,
      isVeryLargeText: false,
    });
    act(() => {
      TestRenderer.create(
        createElement(HookCapture, {
          onValue: (x) => {
            captured = x;
          },
        })
      );
    });
    expect(captured!.minTouchHeight).toBe(Math.round(44 * Math.min(1.2, 1.35)));
  });

  it("bodyLineHeightRatio est 1.3 avec isLargeText", () => {
    let captured: ReturnType<typeof useResponsiveTokens> | undefined;
    mockUseAccessibilityScale.mockReturnValue({
      fontScale: 1,
      isLargeText: true,
      isVeryLargeText: false,
    });
    act(() => {
      TestRenderer.create(
        createElement(HookCapture, {
          onValue: (x) => {
            captured = x;
          },
        })
      );
    });
    expect(captured!.bodyLineHeightRatio).toBe(1.3);
    expect(captured!.headingLineHeightRatio).toBe(1.28);
  });

  it("bodyLineHeightRatio est 1.25 sans isLargeText", () => {
    let captured: ReturnType<typeof useResponsiveTokens> | undefined;
    mockUseAccessibilityScale.mockReturnValue({
      fontScale: 1,
      isLargeText: false,
      isVeryLargeText: false,
    });
    act(() => {
      TestRenderer.create(
        createElement(HookCapture, {
          onValue: (x) => {
            captured = x;
          },
        })
      );
    });
    expect(captured!.bodyLineHeightRatio).toBe(1.25);
    expect(captured!.headingLineHeightRatio).toBe(1.22);
  });

  it("expose les gaps sémantiques alignés sur spacing et landing.cardPadding", () => {
    let captured: ReturnType<typeof useResponsiveTokens> | undefined;
    mockUseAccessibilityScale.mockReturnValue({
      fontScale: 1,
      isLargeText: false,
      isVeryLargeText: false,
    });
    act(() => {
      TestRenderer.create(
        createElement(HookCapture, {
          onValue: (x) => {
            captured = x;
          },
        })
      );
    });
    expect(captured!.pageGap).toBe(captured!.spacingLg);
    expect(captured!.sectionGap).toBe(captured!.spacingMd);
    expect(captured!.fieldGap).toBe(captured!.spacingSm);
    expect(captured!.cardPadding).toBe(captured!.landing.cardPadding);
    expect(captured!.bodyFontSize).toBeGreaterThan(0);
    expect(captured!.buttonFontSize).toBeGreaterThan(0);
    expect(captured!.radiusLg).toBeGreaterThanOrEqual(captured!.radiusMd);
  });
});
