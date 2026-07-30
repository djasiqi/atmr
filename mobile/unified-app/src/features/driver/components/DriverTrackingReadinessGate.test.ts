/**
 * Vérifie que la checklist de la Readiness Gate reflète l'état réel du device
 * (permissions Location + GPS) et n'est PAS court-circuitée par le feature flag
 * `tracking_background_enabled`.
 */
import { Platform } from "react-native";
import * as Location from "expo-location";

import { evaluateTrackingReadiness } from "./DriverTrackingReadinessGate";
import * as battery from "../services/batteryOptimization";
import * as oemPersistence from "../services/oemGuidancePersistence";

jest.mock("expo-location", () => ({
  __esModule: true,
  getForegroundPermissionsAsync: jest.fn(),
  getBackgroundPermissionsAsync: jest.fn(),
  hasServicesEnabledAsync: jest.fn(),
  requestBackgroundPermissionsAsync: jest.fn(),
}));

jest.mock("../services/batteryOptimization", () => ({
  __esModule: true,
  checkBatteryOptimizationStatus: jest.fn(),
  getOemBatteryGuidance: jest.fn(() => ({
    oem: null,
    hasOemSettings: false,
    manufacturer: null,
  })),
  openOemBatterySettings: jest.fn(),
  requestIgnoreBatteryOptimizations: jest.fn(),
}));

jest.mock("../services/oemGuidancePersistence", () => ({
  __esModule: true,
  isOemGuidanceAcknowledgedFor: jest.fn(async () => false),
  markOemGuidanceAcknowledged: jest.fn(),
}));

jest.mock("expo-notifications", () => ({
  getPermissionsAsync: jest.fn(async () => ({ granted: true, status: "granted" })),
  requestPermissionsAsync: jest.fn(),
}));

const fg = Location.getForegroundPermissionsAsync as unknown as jest.Mock;
const bg = Location.getBackgroundPermissionsAsync as unknown as jest.Mock;
const services = Location.hasServicesEnabledAsync as unknown as jest.Mock;
const batteryStatus =
  battery.checkBatteryOptimizationStatus as unknown as jest.Mock;
const oemAck = oemPersistence.isOemGuidanceAcknowledgedFor as unknown as jest.Mock;
const oemGuidance = battery.getOemBatteryGuidance as unknown as jest.Mock;

const originalOs = Platform.OS;

beforeEach(() => {
  fg.mockReset();
  bg.mockReset();
  services.mockReset();
  batteryStatus.mockReset();
  oemAck.mockReset();
  oemAck.mockResolvedValue(false);
  oemGuidance.mockReturnValue({ oem: null, hasOemSettings: false, manufacturer: null });
  Object.defineProperty(Platform, "OS", { configurable: true, value: "android" });
});

afterEach(() => {
  Object.defineProperty(Platform, "OS", { configurable: true, value: originalOs });
});

describe("evaluateTrackingReadiness — device state reflects OS, not feature flag", () => {
  it("permission BG accordée + GPS + précision fine => ready", async () => {
    fg.mockResolvedValue({
      granted: true,
      status: "granted",
      android: { accuracy: "fine" },
    });
    bg.mockResolvedValue({ granted: true, status: "granted" });
    services.mockResolvedValue(true);
    batteryStatus.mockResolvedValue({ checked: true, isIgnoring: true });

    const snapshot = await evaluateTrackingReadiness();

    expect(snapshot.fgPermissionGranted).toBe(true);
    expect(snapshot.bgPermissionGranted).toBe(true);
    expect(snapshot.gpsEnabled).toBe(true);
    expect(snapshot.locationAccuracy).toBe("precise");
    expect(snapshot.batteryStatus).toBe("exempt");
    expect(snapshot.batteryExempt).toBe(true);
    expect(snapshot.ready).toBe(true);
  });

  it("permission BG refusée => bgPermissionGranted=false", async () => {
    fg.mockResolvedValue({
      granted: true,
      android: { accuracy: "fine" },
    });
    bg.mockResolvedValue({ granted: false, status: "denied" });
    services.mockResolvedValue(true);
    batteryStatus.mockResolvedValue({ checked: true, isIgnoring: true });

    const snapshot = await evaluateTrackingReadiness();

    expect(snapshot.bgPermissionGranted).toBe(false);
    expect(snapshot.ready).toBe(false);
  });

  it("GPS désactivé => gpsEnabled=false", async () => {
    fg.mockResolvedValue({
      granted: true,
      android: { accuracy: "fine" },
    });
    bg.mockResolvedValue({ granted: true });
    services.mockResolvedValue(false);
    batteryStatus.mockResolvedValue({ checked: true, isIgnoring: true });

    const snapshot = await evaluateTrackingReadiness();

    expect(snapshot.gpsEnabled).toBe(false);
    expect(snapshot.ready).toBe(false);
  });

  it("localisation approximative => non prêt", async () => {
    fg.mockResolvedValue({
      granted: true,
      android: { accuracy: "coarse" },
    });
    bg.mockResolvedValue({ granted: true });
    services.mockResolvedValue(true);
    batteryStatus.mockResolvedValue({ checked: true, isIgnoring: true });

    const snapshot = await evaluateTrackingReadiness();

    expect(snapshot.locationAccuracy).toBe("approximate");
    expect(snapshot.ready).toBe(false);
  });

  it("batterie indéterminée => unknown, ready non bloqué par batterie", async () => {
    fg.mockResolvedValue({
      granted: true,
      android: { accuracy: "fine" },
    });
    bg.mockResolvedValue({ granted: true });
    services.mockResolvedValue(true);
    batteryStatus.mockResolvedValue({ checked: false, isIgnoring: null });

    const snapshot = await evaluateTrackingReadiness();

    expect(snapshot.batteryStatus).toBe("unknown");
    expect(snapshot.batteryExempt).toBe(false);
    expect(snapshot.ready).toBe(true);
  });

  it("OEM acquitté pour le même fabricant", async () => {
    fg.mockResolvedValue({
      granted: true,
      android: { accuracy: "fine" },
    });
    bg.mockResolvedValue({ granted: true });
    services.mockResolvedValue(true);
    batteryStatus.mockResolvedValue({ checked: true, isIgnoring: false });
    oemGuidance.mockReturnValue({
      oem: "samsung",
      hasOemSettings: true,
      manufacturer: "samsung",
    });
    oemAck.mockResolvedValue(true);

    const snapshot = await evaluateTrackingReadiness();

    expect(snapshot.hasOemSettings).toBe(true);
    expect(snapshot.oemGuidanceAcknowledged).toBe(true);
    expect(snapshot.batteryStatus).toBe("restricted");
    expect(snapshot.ready).toBe(false);
  });

  it("ne tombe pas en panique si expo-location lève (rare hors expo)", async () => {
    fg.mockRejectedValue(new Error("native module missing"));
    bg.mockRejectedValue(new Error("native module missing"));
    services.mockRejectedValue(new Error("native module missing"));
    batteryStatus.mockResolvedValue({ checked: false, isIgnoring: null });

    const snapshot = await evaluateTrackingReadiness();

    expect(snapshot.fgPermissionGranted).toBe(false);
    expect(snapshot.bgPermissionGranted).toBe(false);
    expect(snapshot.gpsEnabled).toBe(false);
    expect(snapshot.ready).toBe(false);
  });
});
