import { resolveFleetHealthScore } from "./healthScoreModel";

describe("resolveFleetHealthScore", () => {
  it("penalizes offline realtime", () => {
    const online = resolveFleetHealthScore({
      delayedCount: 0,
      urgentCount: 0,
      unassignedCount: 0,
      criticalEtaCount: 0,
      realtimeStatus: "healthy",
      policyFailureCount: 0,
      interactionBurstPerMinute: 0,
    });
    const offline = resolveFleetHealthScore({
      delayedCount: 0,
      urgentCount: 0,
      unassignedCount: 0,
      criticalEtaCount: 0,
      realtimeStatus: "offline",
      policyFailureCount: 0,
      interactionBurstPerMinute: 0,
    });
    expect(offline).toBeLessThan(online);
    expect(offline).toBeLessThanOrEqual(65);
  });

  it("does not return fixed 100 when operational stress", () => {
    const score = resolveFleetHealthScore({
      delayedCount: 5,
      urgentCount: 2,
      unassignedCount: 3,
      criticalEtaCount: 1,
      realtimeStatus: "healthy",
      policyFailureCount: 2,
      interactionBurstPerMinute: 0,
    });
    expect(score).toBeLessThan(100);
  });
});
