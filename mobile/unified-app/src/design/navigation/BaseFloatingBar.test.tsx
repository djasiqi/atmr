import { describe, expect, it, jest } from "@jest/globals";
import { createElement } from "react";
import TestRenderer, { act } from "react-test-renderer";
import { View } from "react-native";
import {
  FloatingBarMetricsProvider,
  computeFloatingBarFallbackClearance,
} from "./floatingBarMetrics";

jest.mock("../responsive/useResponsiveTokens", () => ({
  useResponsiveTokens: () => ({
    spacingXs: 4,
    spacingSm: 8,
  }),
}));

const { BaseFloatingBar } = require("./BaseFloatingBar");

describe("BaseFloatingBar", () => {
  it("n’utilise pas position absolute pour la pilule", () => {
    let tree: TestRenderer.ReactTestRenderer;
    act(() => {
      tree = TestRenderer.create(
        createElement(
          FloatingBarMetricsProvider,
          { preset: "driver", bottomPadding: 16 },
          createElement(
            BaseFloatingBar,
            {
              paddingBottom: 16,
              maxBarWidth: 360,
              horizontalPadding: 16,
              preset: "company",
              minInnerHeight: 56,
              isLargeText: false,
            },
            createElement(View, { testID: "child" })
          )
        )
      );
    });
    const views = tree!.root.findAllByType(View);
    const absolutePill = views.find(
      (v) => v.props.style && !Array.isArray(v.props.style) && v.props.style.position === "absolute"
    );
    expect(absolutePill).toBeUndefined();
    const absoluteInArray = views.find(
      (v) =>
        Array.isArray(v.props.style) &&
        v.props.style.some(
          (s: { position?: string } | null) => s && s.position === "absolute"
        )
    );
    expect(absoluteInArray).toBeUndefined();
  });

  it("remonte onLayout via le reporter (fallback clearance cohérent)", () => {
    expect(computeFloatingBarFallbackClearance(56, 20)).toBe(76);
  });
});
