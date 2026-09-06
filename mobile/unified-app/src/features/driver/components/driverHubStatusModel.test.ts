import { describe, expect, it } from "@jest/globals";
import {
  DRIVER_DASHBOARD_STATUS_LINE_HEIGHT,
  collectDriverStatusIssues,
  resolveDriverStatusAreaView,
  type DriverStatusIssueFlags,
} from "./driverHubStatusModel";

const base: DriverStatusIssueFlags = {
  hideTrackingPrepDuplicates: false,
  trackingNeedsAttention: false,
  pushDisclosure: false,
  pushPending: false,
  pushFailed: false,
  pushDenied: false,
  offline: false,
  socketDegraded: false,
  gpsDisabled: false,
  batteryOptimization: false,
  oemRequired: false,
  sessionError: false,
};

describe("driverHubStatusModel", () => {
  it("pas de réserve 48 px — une ligne de statut", () => {
    expect(DRIVER_DASHBOARD_STATUS_LINE_HEIGHT).toBe(12);
    expect(DRIVER_DASHBOARD_STATUS_LINE_HEIGHT).toBeLessThan(24);
  });

  it("zone vide si aucun problème", () => {
    expect(resolveDriverStatusAreaView(collectDriverStatusIssues(base))).toEqual({
      mode: "empty",
      count: 0,
    });
  });

  it("un seul bandeau si un problème", () => {
    const issues = collectDriverStatusIssues({ ...base, offline: true });
    const view = resolveDriverStatusAreaView(issues);
    expect(view.mode).toBe("single");
    expect(view.count).toBe(1);
    if (view.mode === "single") expect(view.issue.id).toBe("offline");
  });

  it("plusieurs problèmes → résumé, pas empilement", () => {
    const issues = collectDriverStatusIssues({
      ...base,
      offline: true,
      gpsDisabled: true,
      pushDenied: true,
    });
    const view = resolveDriverStatusAreaView(issues);
    expect(issues.length).toBe(3);
    expect(view).toEqual({
      mode: "summary",
      count: 3,
      label: "3 éléments à vérifier",
    });
  });

  it("masque les doublons tracking quand le suivi est à vérifier", () => {
    const issues = collectDriverStatusIssues({
      ...base,
      hideTrackingPrepDuplicates: true,
      trackingNeedsAttention: true,
      gpsDisabled: true,
      pushDisclosure: true,
      batteryOptimization: true,
    });
    expect(issues.map((i) => i.id)).toEqual(["tracking"]);
  });
});
