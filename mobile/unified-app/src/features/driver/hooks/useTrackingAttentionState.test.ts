import { requiresLiveTrackingPermission } from "../services/missionLiveTrackingEligibility";

describe("tracking attention — navigation store-safe", () => {
  it("ASSIGNED / consultation : transitions non live non gardées", () => {
    expect(requiresLiveTrackingPermission("ARRIVED")).toBe(false);
    expect(requiresLiveTrackingPermission("COMPLETED")).toBe(false);
    expect(requiresLiveTrackingPermission("CANCELLED")).toBe(false);
  });

  it("démarrage live uniquement sur EN_ROUTE et IN_PROGRESS", () => {
    expect(requiresLiveTrackingPermission("EN_ROUTE")).toBe(true);
    expect(requiresLiveTrackingPermission("IN_PROGRESS")).toBe(true);
  });
});
