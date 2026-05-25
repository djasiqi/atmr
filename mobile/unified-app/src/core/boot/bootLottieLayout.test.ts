import { bootLottieAssets } from "./bootLottieAssets";
import { computeBootLottieDisplaySize, compositionSizeForSource } from "./bootLottieLayout";

describe("bootLottieLayout", () => {
  it("returns Android Medium composition size", () => {
    expect(compositionSizeForSource(bootLottieAssets.androidMedium)).toEqual({ w: 700, h: 840 });
  });

  it("fills the whole screen on a tall phone (iPhone 12-like)", () => {
    const { width, height } = computeBootLottieDisplaySize(390, 844, bootLottieAssets.iphone1314);
    expect(width).toBe(390);
    expect(height).toBe(844);
  });

  it("fills the whole screen on a wide tablet", () => {
    const { width, height } = computeBootLottieDisplaySize(1024, 768, bootLottieAssets.androidMedium);
    expect(width).toBe(1024);
    expect(height).toBe(768);
  });
});
