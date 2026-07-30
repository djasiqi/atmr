import { describe, expect, it, jest, beforeEach, afterEach } from "@jest/globals";
import { Platform } from "react-native";

import {
  isExpoLocationPermissionGranted,
  resolveLocationAccuracy,
} from "./locationPermissionState";

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

describe("resolveLocationAccuracy", () => {
  const originalOs = Platform.OS;

  afterEach(() => {
    Object.defineProperty(Platform, "OS", { configurable: true, value: originalOs });
  });

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("Android fine => precise", () => {
    Object.defineProperty(Platform, "OS", { configurable: true, value: "android" });
    expect(
      resolveLocationAccuracy({ granted: true, android: { accuracy: "fine" } })
    ).toBe("precise");
  });

  it("Android coarse => approximate", () => {
    Object.defineProperty(Platform, "OS", { configurable: true, value: "android" });
    expect(
      resolveLocationAccuracy({ granted: true, android: { accuracy: "coarse" } })
    ).toBe("approximate");
  });

  it("Android granted sans accuracy => unknown", () => {
    Object.defineProperty(Platform, "OS", { configurable: true, value: "android" });
    expect(resolveLocationAccuracy({ granted: true })).toBe("unknown");
  });

  it("non accordé => unknown", () => {
    Object.defineProperty(Platform, "OS", { configurable: true, value: "android" });
    expect(resolveLocationAccuracy({ granted: false })).toBe("unknown");
  });

  it("iOS sans champ accuracy (Expo actuel) + granted => precise", () => {
    Object.defineProperty(Platform, "OS", { configurable: true, value: "ios" });
    expect(resolveLocationAccuracy({ granted: true, ios: { scope: "whenInUse" } })).toBe(
      "precise"
    );
  });

  it("iOS reduced => approximate si le champ est exposé", () => {
    Object.defineProperty(Platform, "OS", { configurable: true, value: "ios" });
    expect(
      resolveLocationAccuracy({ granted: true, ios: { accuracy: "reduced" } })
    ).toBe("approximate");
  });
});
