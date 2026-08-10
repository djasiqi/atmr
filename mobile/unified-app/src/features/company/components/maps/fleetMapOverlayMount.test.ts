import { Platform } from "react-native";

import {
  resolveMountDriverMarkers,
  resolveMountDynamicOverlays,
  shouldEnableFleetClustering,
} from "./fleetMapOverlayMount";
import {
  shouldDisableMapOverlays,
  shouldHoldMapOverlaysDuringReconnect,
} from "./companyMapNativeOverlayGate";

/**
 * Smoke / invariants New Arch iOS (P0.4) — partie automatisable.
 *
 * QA manuelle (device) :
 * - 10 cold starts iOS avec réseau lent / Socket encore `connecting`
 * - markers PNG Metro visibles, aucune polyline/heatmap/ETA
 * - zéro crash
 */
describe("fleetMapOverlayMount — séparation markers / dynamiques", () => {
  const originalOs = Platform.OS;

  afterEach(() => {
    Object.defineProperty(Platform, "OS", { configurable: true, value: originalOs });
  });

  it("désactive le clustering sur iOS même sans simplifyClustering", () => {
    Object.defineProperty(Platform, "OS", { configurable: true, value: "ios" });
    expect(shouldEnableFleetClustering(false)).toBe(false);
    expect(shouldEnableFleetClustering(true)).toBe(false);
  });

  it("conserve le clustering Android sauf simplifyClustering", () => {
    Object.defineProperty(Platform, "OS", { configurable: true, value: "android" });
    expect(shouldEnableFleetClustering(false)).toBe(true);
    expect(shouldEnableFleetClustering(true)).toBe(false);
  });

  it("iOS : markers montables après settle sans Socket healthy", () => {
    Object.defineProperty(Platform, "OS", { configurable: true, value: "ios" });
    expect(resolveMountDriverMarkers(true, false)).toBe(false);
    expect(resolveMountDriverMarkers(true, true)).toBe(true);
    expect(resolveMountDriverMarkers(false, true)).toBe(false);
  });

  it("iOS connecting : markers OK, dynamiques bloqués (gate Socket)", () => {
    Object.defineProperty(Platform, "OS", { configurable: true, value: "ios" });
    const mapReady = true;
    const iosSettled = true;
    const nativeOverlaysEnabled = false; // Socket encore connecting

    expect(resolveMountDriverMarkers(mapReady, iosSettled)).toBe(true);
    expect(resolveMountDynamicOverlays(mapReady, nativeOverlaysEnabled)).toBe(false);
    expect(shouldDisableMapOverlays("connecting", false)).toBe(true);
  });

  it("iOS reconnecting : gate dynamique hold si déjà enabled ; markers indépendants", () => {
    Object.defineProperty(Platform, "OS", { configurable: true, value: "ios" });
    expect(shouldHoldMapOverlaysDuringReconnect("reconnecting", true)).toBe(true);
    expect(shouldDisableMapOverlays("reconnecting", true)).toBe(false);
    expect(resolveMountDriverMarkers(true, true)).toBe(true);
  });

  it("Android : markers dès mapReady sans settle iOS", () => {
    Object.defineProperty(Platform, "OS", { configurable: true, value: "android" });
    expect(resolveMountDriverMarkers(true, false)).toBe(true);
    expect(resolveMountDynamicOverlays(true, true)).toBe(true);
  });
});
