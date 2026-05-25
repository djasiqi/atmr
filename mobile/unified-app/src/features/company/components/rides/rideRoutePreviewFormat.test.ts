import { describe, expect, it } from "@jest/globals";
import { formatRideDistance, formatRideDuration } from "./rideRoutePreviewFormat";

describe("formatRideDistance", () => {
  it("returns em dash for nullish or non-positive values", () => {
    expect(formatRideDistance(null)).toBe("—");
    expect(formatRideDistance(undefined)).toBe("—");
    expect(formatRideDistance(0)).toBe("—");
    expect(formatRideDistance(NaN)).toBe("—");
    expect(formatRideDistance(-5)).toBe("—");
  });

  it("formats meters under 1 km without decimals", () => {
    expect(formatRideDistance(850)).toBe("850 m");
    expect(formatRideDistance(120)).toBe("120 m");
  });

  it("formats kilometers under 10 km with one decimal and comma separator", () => {
    expect(formatRideDistance(2500)).toBe("2,5 km");
    expect(formatRideDistance(9990)).toBe("10,0 km");
  });

  it("formats kilometers above 10 km without decimals", () => {
    expect(formatRideDistance(18000)).toBe("18 km");
    expect(formatRideDistance(123456)).toBe("123 km");
  });
});

describe("formatRideDuration", () => {
  it("returns em dash for nullish or non-positive values", () => {
    expect(formatRideDuration(null)).toBe("—");
    expect(formatRideDuration(undefined)).toBe("—");
    expect(formatRideDuration(0)).toBe("—");
    expect(formatRideDuration(NaN)).toBe("—");
  });

  it("formats seconds under one hour as minutes", () => {
    expect(formatRideDuration(60)).toBe("1 min");
    expect(formatRideDuration(1440)).toBe("24 min");
    expect(formatRideDuration(3540)).toBe("59 min");
  });

  it("formats one hour cleanly", () => {
    expect(formatRideDuration(3600)).toBe("1 h");
    expect(formatRideDuration(7200)).toBe("2 h");
  });

  it("formats hours and minutes with zero-padded minutes", () => {
    expect(formatRideDuration(3900)).toBe("1 h 05");
    expect(formatRideDuration(4320)).toBe("1 h 12");
    expect(formatRideDuration(10860)).toBe("3 h 01");
  });
});
