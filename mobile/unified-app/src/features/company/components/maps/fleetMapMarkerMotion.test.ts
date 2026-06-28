import {
  DEFAULT_SNAP_DISTANCE_M,
  FLEET_MARKER_MOTION_MIN_MS,
  FLEET_MARKER_MOTION_MAX_MS,
  NOOP_DISTANCE_M,
  STALE_RECORDED_GAP_MS,
  easeSmoothStep,
  interpolateFleetMarkerPosition,
  isValidFleetMapCoordinate,
  resolveFleetMarkerMotionDurationMs,
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

  it("animate sur mouvement court (~20 m) avec durée alignée GPS", () => {
    const to = {
      latitude: genevaA.latitude + 0.00018,
      longitude: genevaA.longitude,
    };
    const prev = 1_000_000;
    const next = prev + 8_000;
    const plan = resolveFleetMarkerMotionPlan({
      from: genevaA,
      to,
      previousRecordedAtMs: prev,
      nextRecordedAtMs: next,
    });
    expect(plan.mode).toBe("animate");
    if (plan.mode === "animate") {
      expect(plan.durationMs).toBeGreaterThanOrEqual(FLEET_MARKER_MOTION_MIN_MS);
      expect(plan.durationMs).toBeLessThanOrEqual(FLEET_MARKER_MOTION_MAX_MS);
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

  it("animate si gap recorded_at < STALE_RECORDED_GAP_MS", () => {
    expect(
      resolveFleetMarkerMotionPlan({
        from: genevaA,
        to: {
          latitude: genevaA.latitude + 0.0002,
          longitude: genevaA.longitude,
        },
        previousRecordedAtMs: 1_000_000,
        nextRecordedAtMs: 1_000_000 + 45_000,
      }).mode
    ).toBe("animate");
  });

  it("snap sur gap recorded_at >= STALE_RECORDED_GAP_MS", () => {
    expect(
      resolveFleetMarkerMotionPlan({
        from: genevaA,
        to: {
          latitude: genevaA.latitude + 0.0002,
          longitude: genevaA.longitude,
        },
        previousRecordedAtMs: 1_000_000,
        nextRecordedAtMs: 1_000_000 + STALE_RECORDED_GAP_MS,
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

  it("interpolation smoothstep au milieu", () => {
    const mid = interpolateFleetMarkerPosition(genevaA, genevaB, 0.5);
    expect(mid.latitude).toBeGreaterThan(genevaA.latitude);
    expect(mid.latitude).toBeLessThan(genevaB.latitude);
    expect(easeSmoothStep(0.5)).toBe(0.5);
  });

  it("durée étirée selon intervalle recorded_at", () => {
    const duration = resolveFleetMarkerMotionDurationMs(
      1_000_000,
      1_000_000 + 10_000,
      null,
      20
    );
    expect(duration).toBeGreaterThanOrEqual(FLEET_MARKER_MOTION_MIN_MS);
  });

  it("snap si coordonnée cible invalide", () => {
    expect(
      resolveFleetMarkerMotionPlan({
        from: genevaA,
        to: { latitude: Number.NaN, longitude: 6.1432 },
      })
    ).toEqual({ mode: "snap" });
  });

  it("snap si coordonnée précédente invalide", () => {
    expect(
      resolveFleetMarkerMotionPlan({
        from: { latitude: Number.NaN, longitude: 6.1432 },
        to: genevaB,
      })
    ).toEqual({ mode: "snap" });
  });

  it("isValidFleetMapCoordinate rejette NaN et hors bornes", () => {
    expect(isValidFleetMapCoordinate(genevaA)).toBe(true);
    expect(isValidFleetMapCoordinate({ latitude: Number.NaN, longitude: 0 })).toBe(false);
    expect(isValidFleetMapCoordinate({ latitude: 91, longitude: 0 })).toBe(false);
    expect(isValidFleetMapCoordinate(null)).toBe(false);
  });
});
