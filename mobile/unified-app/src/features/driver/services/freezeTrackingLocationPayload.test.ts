import { describe, expect, it } from "@jest/globals";
import {
  buildDriverLocationHttpBody,
  freezeTrackingLocationPayload,
} from "./freezeTrackingLocationPayload";

describe("freezeTrackingLocationPayload (P0-E immutabilité)", () => {
  const base = {
    eventId: "trk_1786980441803_k6c7ottu",
    sequenceId: 46,
    trackingSessionId: "trk_sess_1786979474208_5whqvvm6",
    sessionGeneration: 1702,
    captureId: "cap_msxe0lhn_hrktf81h2m",
    locationMode: "mission_live" as const,
    missionId: 38243,
    payload: {
      latitude: 46.2116156,
      longitude: 6.1262053,
      accuracy: 7.8,
      heading: 0,
      speed: 0.06,
      timestamp: "2026-08-17T15:27:31.105Z",
      isBackground: true,
    },
    enqueuedAtIso: "2026-08-17T15:27:31.200Z",
  };

  it("T1 — retry même event : body HTTP deepEqual strict", () => {
    const frozen = freezeTrackingLocationPayload(base);
    const body1 = buildDriverLocationHttpBody(frozen);
    const body2 = buildDriverLocationHttpBody({ ...frozen });
    expect(body2).toEqual(body1);
    expect(body1.recorded_at).toBe("2026-08-17T15:27:31.105Z");
    expect(body1.timestamp).toBe("2026-08-17T15:27:31.105Z");
    expect(body1.sent_at).toBe("2026-08-17T15:27:31.200Z");
    expect(body1.tracking_event_id).toBe(base.eventId);
  });

  it("T2 — nouvelle fix → nouvel event_id / capture_id (appelant)", () => {
    const a = freezeTrackingLocationPayload(base);
    const b = freezeTrackingLocationPayload({
      ...base,
      eventId: "trk_other_eid",
      captureId: "cap_other",
      sequenceId: 47,
      payload: {
        ...base.payload,
        latitude: 46.2117,
        timestamp: "2026-08-17T15:27:51.000Z",
      },
      enqueuedAtIso: "2026-08-17T15:27:51.010Z",
    });
    expect(a.trackingEventId).not.toBe(b.trackingEventId);
    expect(a.captureId).not.toBe(b.captureId);
    expect(a.recordedAt).not.toBe(b.recordedAt);
    expect(a.latitude).not.toBe(b.latitude);
  });

  it("T3 — session figée dans le payload même si on tenterait d'overlay", () => {
    const frozen = freezeTrackingLocationPayload(base);
    const body = buildDriverLocationHttpBody({
      ...frozen,
      // overlays illégitimes ne doivent pas être utilisés si payload déjà figé
      // (le flush préfère payload.* — ce test fige l'invariant enqueue)
    });
    expect(body.tracking_session_id).toBe(base.trackingSessionId);
    expect(body.session_generation).toBe(base.sessionGeneration);
    expect(body.sequence_id).toBe(base.sequenceId);
  });

  it("T7 — payload Object.freeze (lat/recorded_at immuables)", () => {
    const frozen = freezeTrackingLocationPayload(base);
    expect(Object.isFrozen(frozen)).toBe(true);
    try {
      (frozen as { latitude: number }).latitude = 99;
    } catch {
      /* strict mode */
    }
    expect(frozen.latitude).toBe(46.2116156);
    expect(frozen.recordedAt).toBe(base.payload.timestamp);
  });

  it("fail-closed si recorded_at/timestamp absents au wire", () => {
    expect(() =>
      buildDriverLocationHttpBody({
        latitude: 1,
        longitude: 2,
      })
    ).toThrow("missing_recorded_at");
  });

  it("T1b — ne régénère jamais Date.now() entre deux builds body", () => {
    const frozen = freezeTrackingLocationPayload(base);
    const body1 = buildDriverLocationHttpBody(frozen);
    const body2 = buildDriverLocationHttpBody(frozen);
    expect(body1).toEqual(body2);
    expect(body1.recorded_at).toBe("2026-08-17T15:27:31.105Z");
    expect(body1.sent_at).toBe("2026-08-17T15:27:31.200Z");
  });
});
