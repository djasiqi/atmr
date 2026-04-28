import { describe, expect, it } from "@jest/globals";
import { clientQueryKeys } from "./queryKeys";

describe("client query keys", () => {
  it("scopes profile key with context_id", () => {
    expect(clientQueryKeys.profile("client:self")).toEqual(["client-profile", "client:self"]);
  });

  it("scopes bookings key with context_id and filter", () => {
    expect(clientQueryKeys.bookings("client:self", "dashboard-limit-1")).toEqual([
      "client-bookings",
      "client:self",
      "dashboard-limit-1",
    ]);
  });

  it("scopes booking detail key with context_id and booking id", () => {
    expect(clientQueryKeys.bookingDetail("client:self", 42)).toEqual([
      "booking-detail",
      "client:self",
      42,
    ]);
  });
});
