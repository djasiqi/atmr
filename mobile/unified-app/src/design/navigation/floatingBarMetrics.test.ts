import { describe, expect, it } from "@jest/globals";
import {
  computeFloatingBarFallbackClearance,
  computeFloatingBarMetrics,
} from "./floatingBarMetrics";

describe("computeFloatingBarFallbackClearance", () => {
  it("somme innerMinHeight + bottomPadding", () => {
    expect(computeFloatingBarFallbackClearance(56, 24)).toBe(80);
  });

  it("clamp les valeurs négatives / non finies à 0", () => {
    expect(computeFloatingBarFallbackClearance(-10, 20)).toBe(20);
    expect(computeFloatingBarFallbackClearance(56, -5)).toBe(56);
    expect(computeFloatingBarFallbackClearance(Number.NaN, 12)).toBe(12);
    expect(computeFloatingBarFallbackClearance(40, Number.POSITIVE_INFINITY)).toBe(40);
  });
});

describe("computeFloatingBarMetrics", () => {
  it("retourne innerHeight, bottomPadding et clearance cohérents", () => {
    const metrics = computeFloatingBarMetrics(64, 34);
    expect(metrics).toEqual({
      innerHeight: 64,
      bottomPadding: 34,
      clearance: 98,
    });
  });

  it("normalise les entrées invalides", () => {
    expect(computeFloatingBarMetrics(-1, Number.NaN)).toEqual({
      innerHeight: 0,
      bottomPadding: 0,
      clearance: 0,
    });
  });
});
