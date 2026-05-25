import { beforeEach, describe, expect, it } from "@jest/globals";
import {
  getRuntimeStabilityCounter,
  recordNotificationDuplicateDroppedTotal,
  resetRuntimeStabilityMetricsForTests,
} from "./runtimeStabilityMetrics";

jest.mock("./perfKpi", () => ({
  emitPerfKpi: jest.fn(),
}));

describe("runtimeStabilityMetrics", () => {
  beforeEach(() => {
    resetRuntimeStabilityMetricsForTests();
  });

  it("increments duplicate dropped counter", () => {
    recordNotificationDuplicateDroppedTotal({ dedup_key: "event:1" });
    recordNotificationDuplicateDroppedTotal({ dedup_key: "event:1" });
    expect(getRuntimeStabilityCounter("notification_duplicate_dropped_total")).toBe(2);
  });
});
