import { describe, expect, it, beforeEach } from "@jest/globals";
import {
  classifyInvalidateSubKey,
  recordRealtimeNotify,
  resetPerfInstrumentationForTests,
} from "./perfInstrumentation";
import {
  buildPerfInstrumentationReport,
  getPerfInstrumentationBucketCountForTests,
  resetPerfInstrumentationStoreForTests,
} from "./perfInstrumentationStore";
import {
  getPerfInstrumentationTier,
  resetPerfInstrumentationTierForTests,
} from "./perfInstrumentationTier";
import { setPerfActiveContext } from "./perfActiveContext";

describe("perfInstrumentation", () => {
  beforeEach(() => {
    resetPerfInstrumentationForTests();
    resetPerfInstrumentationTierForTests();
    process.env.EXPO_PUBLIC_PERF_INSTRUMENTATION_TIER = "dev";
    setPerfActiveContext({ role: "driver", screen: "driver.test" });
  });

  it("classifies invalidate query keys", () => {
    expect(classifyInvalidateSubKey(["driver", "message-hub", "threads", 1])).toBe("threads");
    expect(classifyInvalidateSubKey(["driver", "message-hub", "unread", 1])).toBe("unread");
    expect(classifyInvalidateSubKey(["driver", "message-hub", "messages", 1, "t1"])).toBe("messages");
  });

  it("records notify duration buckets when tier is dev", () => {
    expect(getPerfInstrumentationTier()).toBe("dev");
    recordRealtimeNotify(12, 3);
    recordRealtimeNotify(4, 2);
    expect(getPerfInstrumentationBucketCountForTests()).toBeGreaterThan(0);
    const report = buildPerfInstrumentationReport(5);
    const notifyRow = report.rows.find((r) => r.category === "notify");
    expect(notifyRow?.count).toBe(2);
    expect(notifyRow?.max_ms).toBe(12);
  });
});
