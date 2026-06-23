import { Platform } from "react-native";

import { resolveFleetRasterMarkerNativeProps } from "./resolveFleetRasterMarkerNativeProps";
import {
  isCompanyTransportStableForMapOverlays,
  shouldDisableMapOverlays,
  shouldHoldMapOverlaysDuringReconnect,
} from "./companyMapNativeOverlayGate";

describe("isCompanyTransportStableForMapOverlays", () => {
  it("accepte uniquement healthy", () => {
    expect(isCompanyTransportStableForMapOverlays("healthy")).toBe(true);
    expect(isCompanyTransportStableForMapOverlays("connecting")).toBe(false);
    expect(isCompanyTransportStableForMapOverlays("reconnecting")).toBe(false);
    expect(isCompanyTransportStableForMapOverlays("failed")).toBe(false);
    expect(isCompanyTransportStableForMapOverlays("idle")).toBe(false);
  });
});

describe("overlay gate iOS", () => {
  const originalOs = Platform.OS;

  beforeEach(() => {
    Object.defineProperty(Platform, "OS", { configurable: true, value: "ios" });
  });

  afterEach(() => {
    Object.defineProperty(Platform, "OS", { configurable: true, value: originalOs });
  });

  it("conserve les overlays pendant reconnecting si déjà affichés", () => {
    expect(shouldHoldMapOverlaysDuringReconnect("reconnecting", true)).toBe(true);
    expect(shouldHoldMapOverlaysDuringReconnect("reconnecting", false)).toBe(false);
    expect(shouldHoldMapOverlaysDuringReconnect("connecting", true)).toBe(false);
  });

  it("ne désactive pas pendant reconnecting si overlays déjà visibles", () => {
    expect(shouldDisableMapOverlays("healthy", true)).toBe(false);
    expect(shouldDisableMapOverlays("reconnecting", true)).toBe(false);
    expect(shouldDisableMapOverlays("connecting", true)).toBe(true);
    expect(shouldDisableMapOverlays("failed", true)).toBe(true);
  });
});

describe("resolveFleetRasterMarkerNativeProps", () => {
  it("préfère icon + module Metro quand disponible", () => {
    const moduleId = 42;
    expect(
      resolveFleetRasterMarkerNativeProps({
        uri: "https://cdn.example.com/pin.png",
        width: 24,
        height: 28,
        assetModule: moduleId,
      })
    ).toEqual({ icon: moduleId });
  });

  it("retourne null si uri vide", () => {
    expect(
      resolveFleetRasterMarkerNativeProps({
        uri: "  ",
        width: 24,
        height: 28,
      })
    ).toBeNull();
  });
});
