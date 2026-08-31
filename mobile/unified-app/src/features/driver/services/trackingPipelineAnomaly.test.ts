import {
  __resetPipelineAnomalyForTests,
  evaluatePipelineAnomaly,
  shouldPipelineBeTracked,
} from "./trackingPipelineAnomaly";
import type { TrackingPipelineSnapshot } from "./trackingPipelineObservability";

jest.mock("../../../core/featureFlags/registry", () => ({
  isFeatureEnabled: (key: string) => key === "tracking_pipeline_remote_observability_enabled",
}));

function brokenPipeline(
  overrides: Partial<TrackingPipelineSnapshot> = {}
): TrackingPipelineSnapshot {
  return {
    pipeline_snapshot_version: 1,
    desired_mode: "availability_presence",
    mission_id: null,
    tracking_required: true,
    is_available: true,
    bridge_last_fix_age_s: 5,
    bg_task_last_invoke_age_s: 5,
    watch_callback_age_s: 5,
    j1_handler_age_s: 5,
    j3_accepted_age_s: 5,
    j3_last_result: "accepted",
    j3_last_reject_reason: null,
    queue_last_enqueue_age_s: 5,
    queue_depth: 0,
    queue_head_age_s: null,
    flush_last_attempt_age_s: 10,
    flush_last_sent_age_s: 10,
    durable_ack_age_s: 500,
    owner_present: true,
    owner_generation: "gen-1",
    background_task_registered: true,
    app_state: "active",
    platform: "android",
    app_version: "1.0.0",
    native_build_version: "100",
    runtime_version: "1.0.0",
    ota_update_id: "ota-test",
    last_recovery_reason: null,
    last_recovery_age_s: null,
    recovery_count_15m: 0,
    tracking_runtime_age_s: 300,
    first_suspect: "ACK",
    ...overrides,
  };
}

describe("trackingPipelineAnomaly", () => {
  beforeEach(() => {
    __resetPipelineAnomalyForTests();
  });

  it("shouldPipelineBeTracked — ACK stale + runtime > 120s", () => {
    expect(shouldPipelineBeTracked(brokenPipeline())).toBe(true);
  });

  it("shouldPipelineBeTracked — ACK jamais observé via runtime age", () => {
    expect(
      shouldPipelineBeTracked(
        brokenPipeline({ durable_ack_age_s: null, tracking_runtime_age_s: 200 })
      )
    ).toBe(true);
  });

  it("premier ANOMALY au franchissement puis cooldown 5 min", () => {
    const pipeline = brokenPipeline();
    const t0 = 1_000_000;
    expect(evaluatePipelineAnomaly(pipeline, t0)).toBe("ANOMALY");
    expect(evaluatePipelineAnomaly(pipeline, t0 + 60_000)).toBeNull();
    expect(evaluatePipelineAnomaly(pipeline, t0 + 5 * 60_000)).toBe("ANOMALY");
  });

  it("RECOVERED quand ACK redevient frais", () => {
    const pipeline = brokenPipeline();
    const t0 = 2_000_000;
    expect(evaluatePipelineAnomaly(pipeline, t0)).toBe("ANOMALY");
    const healed = brokenPipeline({ durable_ack_age_s: 5, tracking_runtime_age_s: 300 });
    expect(evaluatePipelineAnomaly(healed, t0 + 10_000)).toBe("RECOVERED");
  });

  it("nouveau snapshot immédiat si first_suspect change", () => {
    const pipeline = brokenPipeline({ first_suspect: "ACK" });
    const t0 = 3_000_000;
    expect(evaluatePipelineAnomaly(pipeline, t0)).toBe("ANOMALY");
    const shifted = brokenPipeline({
      first_suspect: "FLUSH",
      flush_last_attempt_age_s: 500,
      durable_ack_age_s: 500,
      queue_last_enqueue_age_s: 5,
    });
    expect(evaluatePipelineAnomaly(shifted, t0 + 30_000)).toBe("ANOMALY");
  });
});
