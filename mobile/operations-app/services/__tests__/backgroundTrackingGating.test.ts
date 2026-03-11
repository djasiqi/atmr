/**
 * Unit tests for shouldRunBackgroundTracking — Phase 1: mission active only.
 */
import {
  shouldRunBackgroundTracking,
  deriveStartContract,
  satisfiesStartContract,
  deriveStopContract,
  getFirstStopCondition,
  satisfiesStopContract,
  type BgTrackingInputs,
} from "../backgroundTrackingGating";

function inputs(overrides: Partial<BgTrackingInputs> = {}): BgTrackingInputs {
  return {
    isAuthenticated: true,
    role: "driver",
    hasActiveMission: true,
    fgPermission: "granted",
    bgPermission: "granted",
    killSwitchEnabled: false,
    ...overrides,
  };
}

describe("shouldRunBackgroundTracking", () => {
  it("returns true when all conditions met", () => {
    expect(shouldRunBackgroundTracking(inputs())).toBe(true);
  });

  it("returns false when kill switch enabled (priorité absolue)", () => {
    expect(shouldRunBackgroundTracking(inputs({ killSwitchEnabled: true }))).toBe(false);
  });

  it("returns false when role is enterprise", () => {
    expect(shouldRunBackgroundTracking(inputs({ role: "enterprise" }))).toBe(false);
  });

  it("returns false when not authenticated", () => {
    expect(shouldRunBackgroundTracking(inputs({ isAuthenticated: false }))).toBe(false);
  });

  it("returns false when no active mission", () => {
    expect(shouldRunBackgroundTracking(inputs({ hasActiveMission: false }))).toBe(false);
  });

  it("returns false when foreground permission not granted", () => {
    expect(shouldRunBackgroundTracking(inputs({ fgPermission: "denied" }))).toBe(false);
    expect(shouldRunBackgroundTracking(inputs({ fgPermission: "undetermined" }))).toBe(false);
  });

  it("returns false when background permission not granted", () => {
    expect(shouldRunBackgroundTracking(inputs({ bgPermission: "denied" }))).toBe(false);
    expect(shouldRunBackgroundTracking(inputs({ bgPermission: "undetermined" }))).toBe(false);
  });
});

describe("StartContract", () => {
  it("satisfies when all conditions met and not already started", () => {
    const contract = deriveStartContract(inputs(), true);
    expect(satisfiesStartContract(contract)).toBe(true);
  });

  it("does not satisfy when already started", () => {
    const contract = deriveStartContract(inputs(), false);
    expect(satisfiesStartContract(contract)).toBe(false);
  });

  it("does not satisfy when kill switch enabled", () => {
    const contract = deriveStartContract(inputs({ killSwitchEnabled: true }), true);
    expect(satisfiesStartContract(contract)).toBe(false);
  });
});

describe("StopContract", () => {
  it("kill_switch has priority absolue", () => {
    const contract = deriveStopContract(
      inputs({ killSwitchEnabled: true, hasActiveMission: true })
    );
    expect(getFirstStopCondition(contract)).toBe("kill_switch");
    expect(satisfiesStopContract(contract)).toBe(true);
  });

  it("permission_revoked when fg or bg denied", () => {
    const contract = deriveStopContract(inputs({ fgPermission: "denied" }));
    expect(getFirstStopCondition(contract)).toBe("permission_revoked");
  });

  it("mission_ended when no active mission", () => {
    const contract = deriveStopContract(inputs({ hasActiveMission: false }));
    expect(getFirstStopCondition(contract)).toBe("mission_ended");
  });

  it("logout when not authenticated", () => {
    const contract = deriveStopContract(inputs({ isAuthenticated: false }));
    expect(getFirstStopCondition(contract)).toBe("logout");
  });

  it("returns null when no stop condition", () => {
    const contract = deriveStopContract(inputs());
    expect(getFirstStopCondition(contract)).toBe(null);
    expect(satisfiesStopContract(contract)).toBe(false);
  });
});
