import { describe, expect, it } from "@jest/globals";
import { bootLottieAssets } from "./bootLottieAssets";
import { resolveBootLottieByIosModelId } from "./iosBootLottieMapping";

describe("resolveBootLottieByIosModelId", () => {
  it("mappe les codes machine récents (17, Air, 16 Plus)", () => {
    expect(resolveBootLottieByIosModelId("iPhone18,3")).toBe(bootLottieAssets.iphone17);
    expect(resolveBootLottieByIosModelId("iPhone18,4")).toBe(bootLottieAssets.iphoneAir);
    expect(resolveBootLottieByIosModelId("iPhone17,4")).toBe(bootLottieAssets.iphone16Plus);
    expect(resolveBootLottieByIosModelId("iPhone17,2")).toBe(bootLottieAssets.iphone1617ProMax);
    expect(resolveBootLottieByIosModelId("iPhone14,6")).toBe(bootLottieAssets.iphoneSE);
    expect(resolveBootLottieByIosModelId("iPhone14,4")).toBe(bootLottieAssets.iphone13Mini);
  });

  it("retourne null si inconnu ou vide", () => {
    expect(resolveBootLottieByIosModelId(null)).toBeNull();
    expect(resolveBootLottieByIosModelId("")).toBeNull();
    expect(resolveBootLottieByIosModelId("iPhone99,99")).toBeNull();
  });
});
