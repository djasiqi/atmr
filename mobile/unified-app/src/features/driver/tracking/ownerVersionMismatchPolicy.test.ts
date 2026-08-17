import { describe, expect, it } from "@jest/globals";
import { decideOwnerVersionMismatchAction } from "./ownerVersionMismatchPolicy";

describe("D5 owner_version_mismatch policy", () => {
  const prior = { missionId: 38243, missionContextVersion: 1 };
  const desiredSameMission = { missionId: 38243, missionContextVersion: 2 };
  const desiredOtherMission = { missionId: 99999, missionContextVersion: 1 };

  it("T9 : mismatch version même mission → L1 reconcile (pas de STOP)", () => {
    const d = decideOwnerVersionMismatchAction({
      platform: "android",
      taskStarted: true,
      priorOwner: prior,
      desiredOwner: desiredSameMission,
    });
    expect(d.action).toBe("l1_reconcile");
    expect(d.detail).toBe("same_mission_version_bump");
  });

  it("T10 : task non démarrée → abort (rien à Unregister)", () => {
    const d = decideOwnerVersionMismatchAction({
      platform: "android",
      taskStarted: false,
      priorOwner: prior,
      desiredOwner: desiredOtherMission,
    });
    expect(d.action).toBe("abort");
  });

  it("T11 : missionId changé + taskStarted → owned_stop_then_start", () => {
    const d = decideOwnerVersionMismatchAction({
      platform: "android",
      taskStarted: true,
      priorOwner: prior,
      desiredOwner: desiredOtherMission,
    });
    expect(d.action).toBe("owned_stop_then_start");
    expect(d.detail).toBe("mission_id_changed");
  });

  it("owners égaux → abort", () => {
    const d = decideOwnerVersionMismatchAction({
      platform: "android",
      taskStarted: true,
      priorOwner: prior,
      desiredOwner: { ...prior },
    });
    expect(d.action).toBe("abort");
    expect(d.detail).toBe("owners_equal");
  });

  it("iOS → abort", () => {
    const d = decideOwnerVersionMismatchAction({
      platform: "ios",
      taskStarted: true,
      priorOwner: prior,
      desiredOwner: desiredOtherMission,
    });
    expect(d.action).toBe("abort");
  });
});
