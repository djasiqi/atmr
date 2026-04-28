import { emitDriverTelemetry } from "../../core/observability/driverTelemetry";

export async function authenticateDriverBiometric(): Promise<boolean> {
  try {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const localAuth = require("expo-local-authentication");
    const hasHardware = await localAuth.hasHardwareAsync();
    if (!hasHardware) return false;
    const isEnrolled = await localAuth.isEnrolledAsync();
    if (!isEnrolled) return false;
    const result = await localAuth.authenticateAsync({
      promptMessage: "Authenticate to continue driver session",
      disableDeviceFallback: false,
    });
    return Boolean(result.success);
  } catch (error) {
    emitDriverTelemetry("driver.biometric.unavailable", {
      source: "driver.biometric",
      reason: error instanceof Error ? error.message : "biometric_unavailable",
    });
    return false;
  }
}
