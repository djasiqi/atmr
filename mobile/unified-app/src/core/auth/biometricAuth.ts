import { emitDriverTelemetry } from "../observability/driverTelemetry";

export type BiometricAuthOptions = {
  promptMessage?: string;
  cancelLabel?: string;
};

export async function isBiometricAvailable(): Promise<boolean> {
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const localAuth = require("expo-local-authentication");
    const hasHardware = await localAuth.hasHardwareAsync();
    if (!hasHardware) return false;
    return localAuth.isEnrolledAsync();
  } catch {
    return false;
  }
}

export async function authenticateWithBiometric(
  options?: BiometricAuthOptions
): Promise<boolean> {
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const localAuth = require("expo-local-authentication");
    const hasHardware = await localAuth.hasHardwareAsync();
    if (!hasHardware) return false;
    const isEnrolled = await localAuth.isEnrolledAsync();
    if (!isEnrolled) return false;
    const result = await localAuth.authenticateAsync({
      promptMessage: options?.promptMessage ?? "Connexion à Lirie",
      cancelLabel: options?.cancelLabel ?? "Annuler",
      disableDeviceFallback: false,
    });
    return Boolean(result.success);
  } catch (error) {
    emitDriverTelemetry("driver.biometric.unavailable", {
      source: "auth.biometric",
      reason: error instanceof Error ? error.message : "biometric_unavailable",
    });
    return false;
  }
}
