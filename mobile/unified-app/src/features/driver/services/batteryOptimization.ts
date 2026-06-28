/**
 * UX guidée pour Samsung One UI / Android Doze: on détecte si l'app n'est PAS
 * exemptée de l'optimisation batterie (PowerManager.isIgnoringBatteryOptimizations
 * est interrogé par `expo-battery.isBatteryOptimizationEnabledAsync()` côté natif).
 *
 * - Sur Android, on ouvre d'abord la popup système Lirie via
 *   `Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS` (data `package:<id>`).
 * - Puis les écrans fabricant pertinents (Samsung : apps jamais en veille).
 * - En dernier recours : fiche application > Utilisation de la batterie.
 * - On évite `IGNORE_BATTERY_OPTIMIZATION_SETTINGS` (souvent « Gestion de l'utilisation »).
 * - iOS / autres plateformes : no-op silencieux.
 */
import { Platform } from "react-native";

import { emitDriverTelemetry } from "../../../core/observability/driverTelemetry";

export type BatteryOptimizationStatus = {
  /** `true` quand l'app est exemptée (allowlist Doze), `false` sinon, `null` si indéterminable / non supporté. */
  isIgnoring: boolean | null;
  /** `true` quand on a pu interroger l'OS avec succès. */
  checked: boolean;
};

export type OpenBatteryOptimizationResult = {
  /** Intent qui a effectivement été lancé (s'il y en a un). */
  intent: "request_ignore" | "settings" | "oem" | "app_details" | null;
  /** `true` si on a pu déclencher au moins un intent. */
  opened: boolean;
  oem?: string | null;
};

type OemIntentTarget = {
  component: string;
  label: string;
};

const OEM_INTENT_TARGETS: Record<string, OemIntentTarget[]> = {
  xiaomi: [
    {
      component: "com.miui.securitycenter/.permission.AutoStartManagementActivity",
      label: "auto_start",
    },
  ],
  redmi: [
    {
      component: "com.miui.securitycenter/.permission.AutoStartManagementActivity",
      label: "auto_start",
    },
  ],
  huawei: [
    {
      component: "com.huawei.systemmanager/.startupmgr.ui.StartupNormalAppListActivity",
      label: "protected_apps",
    },
  ],
  honor: [
    {
      component: "com.huawei.systemmanager/.startupmgr.ui.StartupNormalAppListActivity",
      label: "protected_apps",
    },
  ],
  oppo: [
    {
      component: "com.coloros.safecenter/.permission.startup.StartupAppListActivity",
      label: "auto_start",
    },
  ],
  realme: [
    {
      component: "com.coloros.safecenter/.permission.startup.StartupAppListActivity",
      label: "auto_start",
    },
  ],
  vivo: [
    {
      component: "com.vivo.permissionmanager/.activity.BgStartUpManagerActivity",
      label: "background_start",
    },
  ],
  samsung: [
    {
      component: "com.samsung.android.lool/com.samsung.android.sm.ui.battery.BatteryActivity",
      label: "device_care",
    },
    {
      component: "com.samsung.android.lool/com.samsung.android.sm.battery.ui.BatteryActivity",
      label: "device_care_alt",
    },
  ],
  oneplus: [
    {
      component: "com.oneplus.security/.chainlaunch.view.ChainLaunchAppListActivity",
      label: "background_activity",
    },
  ],
};

function normalizeManufacturer(value: string | null | undefined): string {
  return String(value || "")
    .trim()
    .toLowerCase();
}

function resolveOemKey(manufacturer: string | null | undefined): string | null {
  const normalized = normalizeManufacturer(manufacturer);
  if (!normalized) return null;
  if (normalized in OEM_INTENT_TARGETS) return normalized;
  for (const key of Object.keys(OEM_INTENT_TARGETS)) {
    if (normalized.includes(key)) return key;
  }
  return null;
}

function readManufacturer(): string | null {
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const Device = require("expo-device") as { manufacturer?: string | null };
    return Device?.manufacturer ? String(Device.manufacturer) : null;
  } catch {
    return null;
  }
}

async function launchOemIntent(target: OemIntentTarget): Promise<boolean> {
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const IntentLauncher = require("expo-intent-launcher") as {
      startActivityAsync: (action: string, params?: Record<string, unknown>) => Promise<unknown>;
    };
    const [pkg, cls] = target.component.split("/");
    await IntentLauncher.startActivityAsync("android.intent.action.MAIN", {
      componentName: `${pkg}/${cls}`,
    });
    return true;
  } catch {
    return false;
  }
}

function getAndroidPackageName(): string | null {
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports -- chargement paresseux pour éviter de cibler iOS / web
    const Application = require("expo-application") as {
      applicationId?: string | null;
    };
    if (Application?.applicationId) return Application.applicationId;
  } catch {
    // fallback ci-dessous
  }
  return null;
}

type IntentLauncherModule = {
  startActivityAsync: (action: string, params?: Record<string, unknown>) => Promise<unknown>;
  ActivityAction?: Record<string, string>;
};

function loadIntentLauncher(): IntentLauncherModule | null {
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports -- chargement paresseux : module Android-only
    return require("expo-intent-launcher") as IntentLauncherModule;
  } catch (error) {
    emitDriverTelemetry("tracking.battery_optimization.unavailable", {
      source: "driver.battery_optimization",
      reason: error instanceof Error ? error.message : "intent_launcher_missing",
    });
    return null;
  }
}

async function launchRequestIgnoreForPackage(
  IntentLauncher: IntentLauncherModule,
  packageName: string
): Promise<OpenBatteryOptimizationResult> {
  const requestAction =
    IntentLauncher.ActivityAction?.REQUEST_IGNORE_BATTERY_OPTIMIZATIONS ??
    "android.settings.REQUEST_IGNORE_BATTERY_OPTIMIZATIONS";
  try {
    await IntentLauncher.startActivityAsync(requestAction, {
      data: `package:${packageName}`,
    });
    return { intent: "request_ignore", opened: true };
  } catch (error) {
    emitDriverTelemetry("tracking.battery_optimization.unavailable", {
      source: "driver.battery_optimization",
      reason: error instanceof Error ? error.message : "request_intent_failed",
      intent: "request_ignore",
    });
    return { intent: null, opened: false };
  }
}

async function launchSamsungNeverSleepingAppsList(
  IntentLauncher: IntentLauncherModule
): Promise<OpenBatteryOptimizationResult> {
  try {
    await IntentLauncher.startActivityAsync(
      "com.samsung.android.sm.ACTION_OPEN_CHECKABLE_LISTACTIVITY",
      {
        packageName: "com.samsung.android.lool",
        extra: { activity_type: 2 },
      }
    );
    return { intent: "oem", opened: true, oem: "samsung" };
  } catch (error) {
    emitDriverTelemetry("tracking.battery_optimization.unavailable", {
      source: "driver.battery_optimization",
      reason: error instanceof Error ? error.message : "samsung_never_sleeping_failed",
      intent: "oem",
      oem: "samsung",
    });
    return { intent: null, opened: false };
  }
}

async function launchAppDetailsSettings(
  IntentLauncher: IntentLauncherModule,
  packageName: string
): Promise<OpenBatteryOptimizationResult> {
  const detailsAction =
    IntentLauncher.ActivityAction?.APPLICATION_DETAILS_SETTINGS ??
    "android.settings.APPLICATION_DETAILS_SETTINGS";
  try {
    await IntentLauncher.startActivityAsync(detailsAction, {
      data: `package:${packageName}`,
    });
    return { intent: "app_details", opened: true };
  } catch (error) {
    emitDriverTelemetry("tracking.battery_optimization.unavailable", {
      source: "driver.battery_optimization",
      reason: error instanceof Error ? error.message : "app_details_intent_failed",
      intent: "app_details",
    });
    return { intent: null, opened: false };
  }
}

/**
 * Ouvre le parcours le plus direct pour autoriser Lirie sans restriction batterie.
 * N'utilise pas la liste globale « Gestion de l'utilisation ».
 */
export async function openDriverBatteryUnrestrictedSettings(): Promise<OpenBatteryOptimizationResult> {
  if (Platform.OS !== "android") {
    return { intent: null, opened: false };
  }

  emitDriverTelemetry("tracking.battery_optimization.user_action", {
    source: "driver.battery_optimization",
    action: "open_unrestricted_settings",
  });

  const IntentLauncher = loadIntentLauncher();
  if (!IntentLauncher) {
    return { intent: null, opened: false };
  }

  const packageName = getAndroidPackageName();
  const oem = resolveOemKey(readManufacturer());

  if (packageName) {
    const requestResult = await launchRequestIgnoreForPackage(IntentLauncher, packageName);
    if (requestResult.opened) return requestResult;
  }

  if (oem === "samsung") {
    const neverSleeping = await launchSamsungNeverSleepingAppsList(IntentLauncher);
    if (neverSleeping.opened) return neverSleeping;
  }

  if (oem) {
    for (const target of OEM_INTENT_TARGETS[oem] || []) {
      const opened = await launchOemIntent(target);
      if (opened) {
        return { intent: "oem", opened: true, oem };
      }
    }
  }

  if (packageName) {
    return launchAppDetailsSettings(IntentLauncher, packageName);
  }

  return { intent: null, opened: false };
}

export function getOemBatteryGuidance(): {
  oem: string | null;
  manufacturer: string | null;
  hasOemSettings: boolean;
} {
  const manufacturer = readManufacturer();
  const oem = resolveOemKey(manufacturer);
  return {
    oem,
    manufacturer,
    hasOemSettings: Boolean(oem),
  };
}

/**
 * Ouvre les réglages avancés du fabricant (auto-start, protected apps, etc.).
 * Fallback chaîné: intent OEM -> popup Doze -> écran générique.
 */
export async function openOemBatterySettings(): Promise<OpenBatteryOptimizationResult> {
  if (Platform.OS !== "android") {
    return { intent: null, opened: false };
  }

  const manufacturer = readManufacturer();
  const oem = resolveOemKey(manufacturer);
  emitDriverTelemetry("tracking.battery_optimization.user_action", {
    source: "driver.battery_optimization",
    action: "open_oem_settings",
    oem: oem || "unknown",
    manufacturer: manufacturer || "unknown",
  });

  if (oem) {
    for (const target of OEM_INTENT_TARGETS[oem] || []) {
      const opened = await launchOemIntent(target);
      if (opened) {
        return { intent: "oem", opened: true, oem };
      }
    }
  }

  const fallback = await openDriverBatteryUnrestrictedSettings();
  return { ...fallback, oem };
}

let lastKnownIsIgnoring: boolean | null = null;
let detectedEventEmittedForCurrentState = false;

/**
 * Interroge l'OS pour savoir si l'app est exemptée de l'optimisation batterie.
 * Sur Android, repose sur `expo-battery.isBatteryOptimizationEnabledAsync()` (qui
 * appelle `PowerManager.isIgnoringBatteryOptimizations` natif). Sur iOS, retourne
 * `{ isIgnoring: null, checked: false }`.
 */
export async function checkBatteryOptimizationStatus(): Promise<BatteryOptimizationStatus> {
  if (Platform.OS !== "android") {
    return { isIgnoring: null, checked: false };
  }
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports -- chargement paresseux : module Android-only
    const Battery = require("expo-battery") as {
      isBatteryOptimizationEnabledAsync?: () => Promise<boolean>;
    };
    if (typeof Battery.isBatteryOptimizationEnabledAsync !== "function") {
      emitDriverTelemetry("tracking.battery_optimization.check_failed", {
        source: "driver.battery_optimization",
        reason: "expo_battery_api_missing",
      });
      return { isIgnoring: null, checked: false };
    }
    const optimizationEnabled = await Battery.isBatteryOptimizationEnabledAsync();
    const isIgnoring = !optimizationEnabled;
    updateIsIgnoringCache(isIgnoring);
    return { isIgnoring, checked: true };
  } catch (error) {
    emitDriverTelemetry("tracking.battery_optimization.check_failed", {
      source: "driver.battery_optimization",
      reason: error instanceof Error ? error.message : "check_threw",
    });
    return { isIgnoring: null, checked: false };
  }
}

function updateIsIgnoringCache(nextIsIgnoring: boolean): void {
  const previous = lastKnownIsIgnoring;
  if (previous === nextIsIgnoring) {
    if (!nextIsIgnoring && !detectedEventEmittedForCurrentState) {
      emitDriverTelemetry("tracking.battery_optimization.detected", {
        source: "driver.battery_optimization",
        is_ignoring: false,
      });
      detectedEventEmittedForCurrentState = true;
    }
    return;
  }
  lastKnownIsIgnoring = nextIsIgnoring;
  if (nextIsIgnoring) {
    detectedEventEmittedForCurrentState = false;
    if (previous === false) {
      emitDriverTelemetry("tracking.battery_optimization.exempted", {
        source: "driver.battery_optimization",
        previously_ignoring: previous,
      });
    }
  } else {
    emitDriverTelemetry("tracking.battery_optimization.detected", {
      source: "driver.battery_optimization",
      is_ignoring: false,
    });
    detectedEventEmittedForCurrentState = true;
  }
}

/**
 * @deprecated Préférer `openDriverBatteryUnrestrictedSettings()`. Ouvre la liste
 * globale qui affiche souvent « Gestion de l'utilisation » sur les OEM récents.
 */
export async function openBatteryOptimizationSettingsScreen(): Promise<OpenBatteryOptimizationResult> {
  if (Platform.OS !== "android") {
    return { intent: null, opened: false };
  }

  emitDriverTelemetry("tracking.battery_optimization.user_action", {
    source: "driver.battery_optimization",
    action: "open_settings_screen",
  });

  let IntentLauncher: {
    startActivityAsync: (action: string, params?: Record<string, unknown>) => Promise<unknown>;
    ActivityAction?: Record<string, string>;
  };
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports -- chargement paresseux : module Android-only
    IntentLauncher = require("expo-intent-launcher");
  } catch (error) {
    emitDriverTelemetry("tracking.battery_optimization.unavailable", {
      source: "driver.battery_optimization",
      reason: error instanceof Error ? error.message : "intent_launcher_missing",
    });
    return { intent: null, opened: false };
  }

  const settingsAction =
    IntentLauncher.ActivityAction?.IGNORE_BATTERY_OPTIMIZATION_SETTINGS ??
    "android.settings.IGNORE_BATTERY_OPTIMIZATION_SETTINGS";

  try {
    await IntentLauncher.startActivityAsync(settingsAction);
    return { intent: "settings", opened: true };
  } catch (error) {
    emitDriverTelemetry("tracking.battery_optimization.unavailable", {
      source: "driver.battery_optimization",
      reason: error instanceof Error ? error.message : "settings_intent_failed",
      intent: "settings",
    });
    return { intent: null, opened: false };
  }
}

/**
 * Demande à l'utilisateur d'exempter l'app (alias de
 * `openDriverBatteryUnrestrictedSettings()`).
 */
export async function requestIgnoreBatteryOptimizations(): Promise<OpenBatteryOptimizationResult> {
  return openDriverBatteryUnrestrictedSettings();
}

/**
 * @deprecated utiliser `openDriverBatteryUnrestrictedSettings()`. Conservé pour compat.
 */
export async function openBatteryOptimizationSettings(): Promise<void> {
  await openDriverBatteryUnrestrictedSettings();
}

/** Reset interne — réservé aux tests. */
export function __resetBatteryOptimizationCacheForTests(): void {
  lastKnownIsIgnoring = null;
  detectedEventEmittedForCurrentState = false;
}
