import {
  PIPELINE_SNAPSHOT_VERSION,
  PIPELINE_STALE_SECONDS,
  __resetTrackingPipelineObservabilityForTests,
  computePipelineFirstSuspect,
  recordPipelineJ1Handler,
  recordPipelineJ3Decision,
} from "./trackingPipelineObservability";

jest.mock("../../../core/featureFlags/registry", () => ({
  isFeatureEnabled: (key: string) => key === "tracking_pipeline_remote_observability_enabled",
}));

describe("trackingPipelineObservability", () => {
  beforeEach(() => {
    __resetTrackingPipelineObservabilityForTests();
  });

  it("computePipelineFirstSuspect — J3 rejeté immédiat", () => {
    expect(
      computePipelineFirstSuspect({
        bridge_last_fix_age_s: 5,
        bg_task_last_invoke_age_s: 5,
        watch_callback_age_s: 5,
        j1_handler_age_s: 5,
        j3_accepted_age_s: 200,
        j3_last_result: "rejected",
        queue_last_enqueue_age_s: null,
        flush_last_attempt_age_s: null,
        durable_ack_age_s: 500,
      })
    ).toBe("J3_GATE");
  });

  it("computePipelineFirstSuspect — ACK stale après flush frais", () => {
    expect(
      computePipelineFirstSuspect({
        bridge_last_fix_age_s: 5,
        bg_task_last_invoke_age_s: 5,
        watch_callback_age_s: 5,
        j1_handler_age_s: 5,
        j3_accepted_age_s: 5,
        j3_last_result: "accepted",
        queue_last_enqueue_age_s: 5,
        flush_last_attempt_age_s: 10,
        durable_ack_age_s: 500,
      })
    ).toBe("ACK");
  });

  it("computePipelineFirstSuspect — BG task stale", () => {
    expect(
      computePipelineFirstSuspect({
        bridge_last_fix_age_s: 5,
        bg_task_last_invoke_age_s: 500,
        watch_callback_age_s: 500,
        j1_handler_age_s: 500,
        j3_accepted_age_s: 500,
        j3_last_result: "unknown",
        queue_last_enqueue_age_s: 500,
        flush_last_attempt_age_s: 500,
        durable_ack_age_s: 500,
      })
    ).toBe("BG_TASK");
  });

  it("enregistre J1/J3 sans effet de bord métier", () => {
    const now = Date.now();
    recordPipelineJ1Handler(now - 4_000);
    recordPipelineJ3Decision({ result: "accepted", nowMs: now - 3_000 });
    expect(PIPELINE_SNAPSHOT_VERSION).toBe(1);
    expect(PIPELINE_STALE_SECONDS).toBe(120);
  });
});
