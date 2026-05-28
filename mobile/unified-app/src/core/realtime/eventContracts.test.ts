import { describe, expect, it } from "@jest/globals";
import { normalizeCompanyEventType } from "./eventContracts";

describe("company realtime event aliases", () => {
  it("maps ride and mission aliases to booking canonical events", () => {
    expect(normalizeCompanyEventType("ride_updated")).toBe("booking_updated");
    expect(normalizeCompanyEventType("mission_updated")).toBe("booking_updated");
    expect(normalizeCompanyEventType("ride_cancelled")).toBe("booking_cancelled");
    expect(normalizeCompanyEventType("mission_cancelled")).toBe("booking_cancelled");
  });

  it("keeps company_dispatch_update as canonical aggregate signal", () => {
    expect(normalizeCompanyEventType("company_dispatch_update")).toBe("company_dispatch_update");
  });

  // Phase 2 PR B/C — gate D3.1
  it("maps dispatch_assignment to its own canonical type", () => {
    expect(normalizeCompanyEventType("dispatch_assignment")).toBe("dispatch_assignment");
  });

  it("maps dispatch_run_started/completed/failed to dispatch_run_lifecycle", () => {
    expect(normalizeCompanyEventType("dispatch_run_started")).toBe("dispatch_run_lifecycle");
    expect(normalizeCompanyEventType("dispatch_run_completed")).toBe("dispatch_run_lifecycle");
    expect(normalizeCompanyEventType("dispatch_run_failed")).toBe("dispatch_run_lifecycle");
  });
});
