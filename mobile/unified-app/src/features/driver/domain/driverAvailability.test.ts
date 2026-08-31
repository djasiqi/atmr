import { describe, expect, it } from "@jest/globals";
import { resolveDriverAvailabilityFromProfile } from "./driverAvailability";

describe("resolveDriverAvailabilityFromProfile", () => {
  it("profil absent → UNKNOWN (null), jamais true par défaut", () => {
    expect(resolveDriverAvailabilityFromProfile(null)).toBeNull();
    expect(resolveDriverAvailabilityFromProfile(undefined)).toBeNull();
    expect(resolveDriverAvailabilityFromProfile({})).toBeNull();
  });

  it("is_available=false (cache / DB) → UNAVAILABLE", () => {
    expect(resolveDriverAvailabilityFromProfile({ is_available: false })).toBe(false);
    expect(resolveDriverAvailabilityFromProfile({ is_available: 0 })).toBe(false);
    expect(resolveDriverAvailabilityFromProfile({ is_available: "false" })).toBe(false);
  });

  it("is_available=true → AVAILABLE", () => {
    expect(resolveDriverAvailabilityFromProfile({ is_available: true })).toBe(true);
    expect(resolveDriverAvailabilityFromProfile({ is_available: 1 })).toBe(true);
    expect(resolveDriverAvailabilityFromProfile({ is_available: "true" })).toBe(true);
  });
});
