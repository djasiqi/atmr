/**
 * Vérifie que la checklist de la Readiness Gate reflète l'état réel du device
 * (permissions Location + GPS) et n'est PAS court-circuitée par le feature flag
 * `tracking_background_enabled` (régression observée en prod : tout affichait
 * faux alors que perm BG = "Toujours" et GPS = activé).
 */
import * as Location from "expo-location";

import { evaluateTrackingReadiness } from "./DriverTrackingReadinessGate";
import * as battery from "../services/batteryOptimization";

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

const fg = Location.getForegroundPermissionsAsync as unknown as jest.Mock;
const bg = Location.getBackgroundPermissionsAsync as unknown as jest.Mock;
const services = Location.hasServicesEnabledAsync as unknown as jest.Mock;
const batteryStatus =
  battery.checkBatteryOptimizationStatus as unknown as jest.Mock;

beforeEach(() => {
  fg.mockReset();
  bg.mockReset();
  services.mockReset();
  batteryStatus.mockReset();
});

describe("evaluateTrackingReadiness — device state reflects OS, not feature flag", () => {
  it("permission BG accordée + GPS activé => snapshot reflète l'état device", async () => {
    fg.mockResolvedValue({ granted: true, status: "granted" });
    bg.mockResolvedValue({ granted: true, status: "granted" });
    services.mockResolvedValue(true);
    batteryStatus.mockResolvedValue({ checked: true, isIgnoring: true });

    const snapshot = await evaluateTrackingReadiness();

    // Le bug d'origine : ces deux valeurs étaient FAUSSES quand
    // EXPO_PUBLIC_ENABLE_BG_LOCATION != "1", peu importe l'état réel du device.
    expect(snapshot.fgPermissionGranted).toBe(true);
    expect(snapshot.bgPermissionGranted).toBe(true);
    expect(snapshot.gpsEnabled).toBe(true);
    expect(snapshot.batteryExempt).toBe(true);
  });

  it("permission BG refusée => bgPermissionGranted=false", async () => {
    fg.mockResolvedValue({ granted: true });
    bg.mockResolvedValue({ granted: false, status: "denied" });
    services.mockResolvedValue(true);
    batteryStatus.mockResolvedValue({ checked: true, isIgnoring: true });

    const snapshot = await evaluateTrackingReadiness();

    expect(snapshot.bgPermissionGranted).toBe(false);
    expect(snapshot.ready).toBe(false);
  });

  it("GPS désactivé => gpsEnabled=false", async () => {
    fg.mockResolvedValue({ granted: true });
    bg.mockResolvedValue({ granted: true });
    services.mockResolvedValue(false);
    batteryStatus.mockResolvedValue({ checked: true, isIgnoring: true });

    const snapshot = await evaluateTrackingReadiness();

    expect(snapshot.gpsEnabled).toBe(false);
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
