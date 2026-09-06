import { describe, expect, it } from "@jest/globals";
import {
  BOOT_BRAND_LOGO_HEIGHT,
  BOOT_BRAND_LOGO_WIDTH,
  SPLASH_BACKGROUND_COLOR,
} from "./bootSurface";

describe("bootSurface", () => {
  it("fixe le fond LIRIE et refuse le blanc", () => {
    expect(SPLASH_BACKGROUND_COLOR).toBe("#EAF3F1");
    expect(SPLASH_BACKGROUND_COLOR.toUpperCase()).not.toBe("#FFFFFF");
  });

  it("aligne le wordmark JS sur le splash natif Expo (imageWidth 220)", () => {
    expect(BOOT_BRAND_LOGO_WIDTH).toBe(220);
    expect(BOOT_BRAND_LOGO_HEIGHT).toBe(95);
  });
});
