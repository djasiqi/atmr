import { beforeEach, describe, expect, it } from "@jest/globals";
import { contextRealtimeRouter } from "./contextRealtimeRouter";

describe("context realtime router", () => {
  beforeEach(() => {
    contextRealtimeRouter.setActiveContext(null);
  });

  it("dispatches events only to matching context listeners", () => {
    const driverEvents: unknown[] = [];
    const companyEvents: unknown[] = [];
    const unsubDriver = contextRealtimeRouter.subscribe("driver:42", (event) => {
      driverEvents.push(event);
    });
    const unsubCompany = contextRealtimeRouter.subscribe("company:7", (event) => {
      companyEvents.push(event);
    });

    contextRealtimeRouter.dispatch("driver:42", { mission_id: 1 });
    contextRealtimeRouter.dispatch("company:7", { booking_id: 99 });

    expect(driverEvents).toEqual([{ mission_id: 1 }]);
    expect(companyEvents).toEqual([{ booking_id: 99 }]);

    unsubDriver();
    unsubCompany();
  });

  it("filters events when active context type does not match", () => {
    const companyEvents: unknown[] = [];
    const unsub = contextRealtimeRouter.subscribe("company:7", (event) => {
      companyEvents.push(event);
    });

    contextRealtimeRouter.setActiveContext("company");
    contextRealtimeRouter.dispatch(
      "company:7",
      { booking_id: 99, context_type: "driver" },
      { contextType: "driver" }
    );
    expect(companyEvents).toEqual([]);

    contextRealtimeRouter.dispatch(
      "company:7",
      { booking_id: 100, context_type: "company" },
      { contextType: "company" }
    );
    expect(companyEvents).toEqual([{ booking_id: 100, context_type: "company" }]);
    unsub();
  });
});
