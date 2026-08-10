import {
  easeSmoothStep,
  interpolateMarkerPosition,
  projectPositionAlongVelocity,
  resolveMarkerMotionDurationMs,
  resolveMotionDurationFromDistance,
  haversineDistanceMeters,
  isApproximateGpsAccuracy,
  MARKER_MOTION_DEFAULT_MS,
  MARKER_MOTION_MIN_MS,
  MARKER_MOTION_MAX_MS,
  MARKER_MOTION_DURATION_STRETCH,
  MARKER_MOTION_PROJECT_FRACTION,
} from '../driverMarkerMotion';

describe('driverMarkerMotion', () => {
  it('canary : dead reckoning désactivé (fraction 0)', () => {
    expect(MARKER_MOTION_PROJECT_FRACTION).toBe(0);
  });

  it('accuracy 80 => approximatif ; 8 => précis (coords inchangées côté helper)', () => {
    expect(isApproximateGpsAccuracy(80)).toBe(true);
    expect(isApproximateGpsAccuracy(8)).toBe(false);
    expect(isApproximateGpsAccuracy(null)).toBe(false);
  });

  it('resolveMarkerMotionDurationMs borne et étire l intervalle', () => {
    const now = 1_000_000;
    expect(resolveMarkerMotionDurationMs(null, now)).toBe(MARKER_MOTION_DEFAULT_MS);
    expect(resolveMarkerMotionDurationMs(now - 500, now)).toBe(MARKER_MOTION_MIN_MS);
    expect(resolveMarkerMotionDurationMs(now - 8000, now)).toBe(
      Math.min(MARKER_MOTION_MAX_MS, 8000 * MARKER_MOTION_DURATION_STRETCH)
    );
  });

  it('easeSmoothStep borne 0 et 1', () => {
    expect(easeSmoothStep(0)).toBe(0);
    expect(easeSmoothStep(1)).toBe(1);
    expect(easeSmoothStep(0.5)).toBeGreaterThan(0.4);
    expect(easeSmoothStep(0.5)).toBeLessThan(0.6);
  });

  it('interpolateMarkerPosition avec smoothstep', () => {
    const from = { lat: 46, lng: 6 };
    const to = { lat: 47, lng: 7 };
    expect(interpolateMarkerPosition(from, to, 0)).toEqual(from);
    expect(interpolateMarkerPosition(from, to, 1)).toEqual(to);
  });

  it('resolveMotionDurationFromDistance prolonge les petits mouvements', () => {
    expect(resolveMotionDurationFromDistance(2000, 5)).toBeGreaterThanOrEqual(3200);
  });

  it('projectPositionAlongVelocity déplace selon la vélocité', () => {
    const next = projectPositionAlongVelocity(46, 6, 0.00001, 0.00002, 1000);
    expect(next.lat).toBeGreaterThan(46);
    expect(next.lng).toBeGreaterThan(6);
  });

  it('haversineDistanceMeters', () => {
    const d = haversineDistanceMeters({ lat: 46.2, lng: 6.1 }, { lat: 46.2, lng: 6.101 });
    expect(d).toBeGreaterThan(50);
    expect(d).toBeLessThan(200);
  });
});
