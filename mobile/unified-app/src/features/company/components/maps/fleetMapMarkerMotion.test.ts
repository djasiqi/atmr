import {
  DEFAULT_SNAP_DISTANCE_M,
  NOOP_DISTANCE_M,
  resolveFleetMarkerMotionPlan,
  shouldApplyFleetMarkerCommit,
} from "./fleetMapMarkerMotion";

const genevaA = { latitude: 46.2044, longitude: 6.1432 };
const genevaB = { latitude: 46.20458, longitude: 6.1434 };
const genevaFar = { latitude: 46.207, longitude: 6.15 };

describe("fleetMapMarkerMotion", () => {
  it("snap sur premier point", () => {
    expect(resolveFleetMarkerMotionPlan({ to: genevaA })).toEqual({ mode: "snap" });
  });

  it("snap si distance <= NOOP_DISTANCE_M", () => {
    const to = {
      latitude: genevaA.latitude + 0.000003,
      longitude: genevaA.longitude + 0.000003,
    };
    expect(
      resolveFleetMarkerMotionPlan({
        from: genevaA,
        to,
        noopDistanceM: NOOP_DISTANCE_M,
      })
    ).toEqual({ mode: "snap" });
  });

  it("animate sur mouvement court (~20 m)", () => {
    const to = {
      latitude: genevaA.latitude + 0.00018,
      longitude: genevaA.longitude,
    };
    const plan = resolveFleetMarkerMotionPlan({ from: genevaA, to });
    expect(plan.mode).toBe("animate");
    if (plan.mode === "animate") {
      expect(plan.durationMs).toBeGreaterThanOrEqual(800);
      expect(plan.durationMs).toBeLessThanOrEqual(1500);
    }
  });

  it("snap sur mouvement long (>= snapDistanceM)", () => {
    expect(
      resolveFleetMarkerMotionPlan({
        from: genevaA,
        to: genevaFar,
        snapDistanceM: DEFAULT_SNAP_DISTANCE_M,
      })
    ).toEqual({ mode: "snap" });
  });

  it("snapDistanceM injectable : 200 m snap à 220 m", () => {
    const to = {
      latitude: genevaA.latitude + 0.002,
      longitude: genevaA.longitude,
    };
    expect(
      resolveFleetMarkerMotionPlan({
        from: genevaA,
        to,
        snapDistanceM: 200,
      }).mode
    ).toBe("snap");
  });

  it("snapDistanceM injectable : animate à 150 m avec seuil 200", () => {
    const to = {
      latitude: genevaA.latitude + 0.00135,
      longitude: genevaA.longitude,
    };
    expect(
      resolveFleetMarkerMotionPlan({
        from: genevaA,
        to,
        snapDistanceM: 200,
      }).mode
    ).toBe("animate");
  });

  it("snap sur gap recorded_at >= 30 s", () => {
    expect(
      resolveFleetMarkerMotionPlan({
        from: genevaA,
        to: {
          latitude: genevaA.latitude + 0.0002,
          longitude: genevaA.longitude,
        },
        previousRecordedAtMs: 1_000_000,
        nextRecordedAtMs: 1_000_000 + 45_000,
      })
    ).toEqual({ mode: "snap" });
  });

  it("snap si location_status stale", () => {
    expect(
      resolveFleetMarkerMotionPlan({
        from: genevaA,
        to: {
          latitude: genevaA.latitude + 0.0002,
          longitude: genevaA.longitude,
        },
        locationStatus: "stale",
      })
    ).toEqual({ mode: "snap" });
  });

  it("snap si markerKeyChanged même à courte distance", () => {
    expect(
      resolveFleetMarkerMotionPlan({
        from: genevaA,
        to: genevaB,
        markerKeyChanged: true,
      })
    ).toEqual({ mode: "snap" });
  });

  it("shouldApplyFleetMarkerCommit : seule la seq courante commit", () => {
    expect(shouldApplyFleetMarkerCommit(2, 2)).toBe(true);
    expect(shouldApplyFleetMarkerCommit(1, 2)).toBe(false);
    expect(shouldApplyFleetMarkerCommit(2, 3)).toBe(false);
  });
});
