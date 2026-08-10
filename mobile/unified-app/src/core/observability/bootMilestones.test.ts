import {
  hasBootMilestone,
  markBootMilestone,
  resetBootMilestonesForTests,
} from "./bootMilestones";
import { setPerfKpiSink } from "./perfKpi";

describe("bootMilestones", () => {
  beforeEach(() => {
    resetBootMilestonesForTests();
    setPerfKpiSink(null);
  });

  it("n’émet chaque jalon qu’une seule fois", () => {
    const events: string[] = [];
    setPerfKpiSink((event, payload) => {
      if (event === "perf.boot.milestone") {
        events.push(String(payload.milestone));
      }
    });

    markBootMilestone("APP_JS_READY");
    markBootMilestone("APP_JS_READY");
    markBootMilestone("MAP_READY");

    expect(events).toEqual(["APP_JS_READY", "MAP_READY"]);
    expect(hasBootMilestone("APP_JS_READY")).toBe(true);
    expect(hasBootMilestone("SOCKET_HEALTHY")).toBe(false);
  });
});
