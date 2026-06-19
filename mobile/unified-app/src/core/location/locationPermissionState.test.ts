import { describe, expect, it } from "@jest/globals";

import { isExpoLocationPermissionGranted } from "./locationPermissionState";

describe("isExpoLocationPermissionGranted", () => {
  it("accepte granted=true", () => {
    expect(isExpoLocationPermissionGranted({ granted: true })).toBe(true);
  });

  it("accepte status=granted même si granted est absent", () => {
    expect(isExpoLocationPermissionGranted({ status: "granted" })).toBe(true);
  });

  it("refuse denied ou undetermined", () => {
    expect(isExpoLocationPermissionGranted({ status: "denied" })).toBe(false);
    expect(isExpoLocationPermissionGranted({ status: "undetermined" })).toBe(false);
    expect(isExpoLocationPermissionGranted(null)).toBe(false);
  });
});
