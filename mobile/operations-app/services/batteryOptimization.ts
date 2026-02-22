import { Platform, NativeModules } from "react-native";
import * as Device from "expo-device";

const HAS_NATIVE_MODULE = !!NativeModules.ExpoIntentLauncher;

let _intentLauncher: typeof import("expo-intent-launcher") | null = null;

async function getIntentLauncher() {
  if (!HAS_NATIVE_MODULE) return null;
  if (_intentLauncher) return _intentLauncher;
  try {
    _intentLauncher = await import("expo-intent-launcher");
    return _intentLauncher;
  } catch {
    return null;
  }
}

export function isSamsungDevice(): boolean {
  if (Platform.OS !== "android") return false;
  const brand = (Device.brand ?? "").toLowerCase();
  return brand === "samsung";
}

export async function checkBatteryOptimization(): Promise<{
  needsExemption: boolean;
  isSamsung: boolean;
}> {
  if (Platform.OS !== "android") {
    return { needsExemption: false, isSamsung: false };
  }
  const samsung = isSamsungDevice();
  try {
    const IL = await getIntentLauncher();
    if (!IL) return { needsExemption: samsung, isSamsung: samsung };
    const result = await IL.startActivityAsync(
      "android.settings.REQUEST_IGNORE_BATTERY_OPTIMIZATIONS",
      {
        data: `package:${Device.modelId ?? "ch.liri.operations"}`,
        extra: { _checkOnly: true },
      }
    );
    const exempted = result.resultCode === IL.ResultCode.Success;
    return { needsExemption: !exempted, isSamsung: samsung };
  } catch {
    return { needsExemption: samsung, isSamsung: samsung };
  }
}

export async function requestBatteryOptimizationExemption(): Promise<boolean> {
  if (Platform.OS !== "android") return true;
  try {
    const IL = await getIntentLauncher();
    if (!IL) return false;
    const packageName = "ch.liri.operations";
    const result = await IL.startActivityAsync(
      "android.settings.REQUEST_IGNORE_BATTERY_OPTIMIZATIONS",
      { data: `package:${packageName}` }
    );
    return result.resultCode === IL.ResultCode.Success;
  } catch {
    return false;
  }
}

export async function openSamsungBatterySettings(): Promise<void> {
  const IL = await getIntentLauncher();
  if (!IL) return;
  try {
    await IL.startActivityAsync(
      IL.ActivityAction.APPLICATION_DETAILS_SETTINGS,
      { data: "package:com.samsung.android.lool" }
    );
  } catch {
    try {
      await IL.startActivityAsync("android.settings.BATTERY_SAVER_SETTINGS");
    } catch {
      await IL.startActivityAsync(IL.ActivityAction.SETTINGS);
    }
  }
}
