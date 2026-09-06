import { describe, expect, it } from "@jest/globals";
import { shouldReleaseNativeSplash } from "./hideNativeSplashWhenReady";

describe("shouldReleaseNativeSplash", () => {
  it("attend que l’overlay soit peint", () => {
    expect(shouldReleaseNativeSplash({ overlayLaidOut: false, overlayWillNeverShow: false })).toBe(
      false
    );
  });

  it("libère après layout de l’overlay", () => {
    expect(shouldReleaseNativeSplash({ overlayLaidOut: true, overlayWillNeverShow: false })).toBe(
      true
    );
  });

  it("libère si l’overlay ne s’affichera jamais (erreur session)", () => {
    expect(shouldReleaseNativeSplash({ overlayLaidOut: false, overlayWillNeverShow: true })).toBe(
      true
    );
  });
});
