import { describe, expect, it } from "@jest/globals";
import { resolveCompanyDeepLink, resolveDriverDeepLink } from "./deepLinkHandler";

describe("resolveDriverDeepLink", () => {
  it("resolves mission deep link", () => {
    expect(resolveDriverDeepLink("atmr://mission/123")).toEqual({
      route: "/(app)/(driver)/missions/123",
      missionId: 123,
    });
  });

  it("resolves quick action deep link", () => {
    expect(resolveDriverDeepLink("atmr://quick-action?missionId=12&action=complete")).toEqual({
      route: "/quick-action?missionId=12&action=complete",
      missionId: 12,
    });
  });

  it("resolves lirie mission deep link", () => {
    expect(resolveDriverDeepLink("lirie://mission/321")).toEqual({
      route: "/(app)/(driver)/missions/321",
      missionId: 321,
    });
  });

  it("resolves chat thread deep link", () => {
    expect(resolveDriverDeepLink("atmr://chat/thread/dispatch")).toEqual({
      route: "/(app)/(driver)/messages/dispatch",
      missionId: null,
    });
  });
});

describe("resolveCompanyDeepLink", () => {
  it("resolves transfer deep link", () => {
    expect(resolveCompanyDeepLink("atmr://transfer/88")).toEqual({
      route: "/(app)/(company)/ride-details?rideId=88",
      rideId: 88,
    });
  });

  it("resolves rides filter deep link", () => {
    expect(resolveCompanyDeepLink("atmr://rides?filter=urgent")).toEqual({
      route: "/(app)/(company)/rides?filter=urgent",
      rideId: null,
    });
  });

  it("resolves lirie transfer deep link", () => {
    expect(resolveCompanyDeepLink("lirie://transfer/45")).toEqual({
      route: "/(app)/(company)/ride-details?rideId=45",
      rideId: 45,
    });
  });

  it("resolves enterprise offer deep link", () => {
    expect(resolveCompanyDeepLink("lirie://enterprise/offers/123?request=456")).toEqual({
      route: "/(app)/(company)/offers/123?request=456",
      rideId: null,
      offerId: 123,
      requestId: 456,
    });
  });
});
