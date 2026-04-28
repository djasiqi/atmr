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
});
