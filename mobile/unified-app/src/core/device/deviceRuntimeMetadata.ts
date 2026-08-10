/**
 * Métadonnées runtime appareil — source unique pour auth, device-health, observabilité.
 *
 * Best-effort : une lecture impossible (model, OS, etc.) ne doit jamais bloquer
 * une connexion. Seul `installationId` (via getStableDeviceId) est obligatoire
 * pour le contrat auth, et il est résolu ailleurs.
 *
 * Sémantique :
 * - deviceName = nom humain OS si dispo (ex. « iPhone de Drin »)
 * - model = modèle matériel (ex. « iPhone 15 Pro »)
 * Jamais Application.applicationName (nom de l'app « Lirie »).
 */
import { Platform } from "react-native";

export type DeviceRuntimeMetadata = {
  platform: "ios" | "android" | "web" | string;
  /** Nom humain fourni par l'OS (pas le nom de l'application). */
  deviceName: string | null;
  manufacturer: string | null;
  model: string | null;
  deviceType: "phone" | "tablet" | "desktop" | "tv" | "unknown";
  osVersion: string | null;
  appVersion: string | null;
  appBuild: string | null;
  expoRuntimeVersion: string | null;
  otaUpdateId: string | null;
  releaseChannel: string | null;
  releaseSha: string | null;
};

type ExpoDeviceModule = {
  manufacturer?: string | null;
  modelName?: string | null;
  deviceName?: string | null;
  osVersion?: string | null;
  deviceType?: number | null;
  DeviceType?: {
    UNKNOWN?: number;
    PHONE?: number;
    TABLET?: number;
    DESKTOP?: number;
    TV?: number;
  };
};

type ExpoApplicationModule = {
  nativeApplicationVersion?: string | null;
  nativeBuildVersion?: string | null;
  applicationName?: string | null;
};

const APP_LABELS = new Set(["lirie", "atmr", "expo", ""]);

function mapDeviceType(Device: ExpoDeviceModule): DeviceRuntimeMetadata["deviceType"] {
  const raw = Device.deviceType;
  const enumMap = Device.DeviceType;
  if (raw == null || !enumMap) return "unknown";
  if (enumMap.PHONE != null && raw === enumMap.PHONE) return "phone";
  if (enumMap.TABLET != null && raw === enumMap.TABLET) return "tablet";
  if (enumMap.DESKTOP != null && raw === enumMap.DESKTOP) return "desktop";
  if (enumMap.TV != null && raw === enumMap.TV) return "tv";
  return "unknown";
}

function readDeviceModule(): ExpoDeviceModule | null {
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    return require("expo-device") as ExpoDeviceModule;
  } catch {
    return null;
  }
}

function readApplicationModule(): ExpoApplicationModule | null {
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    return require("expo-application") as ExpoApplicationModule;
  } catch {
    return null;
  }
}

function isAppLabel(value: string | null | undefined): boolean {
  return APP_LABELS.has((value || "").trim().toLowerCase());
}

/**
 * Collecte best-effort des métadonnées appareil / app / OS.
 * Ne lève jamais : chaque champ peut être null.
 */
export function resolveDeviceRuntimeMetadata(): DeviceRuntimeMetadata {
  const Device = readDeviceModule();
  const Application = readApplicationModule();

  let manufacturer: string | null = null;
  let model: string | null = null;
  let deviceName: string | null = null;
  let osVersion: string | null = null;
  let deviceType: DeviceRuntimeMetadata["deviceType"] = "unknown";

  if (Device) {
    manufacturer = Device.manufacturer ? String(Device.manufacturer) : null;
    model = Device.modelName ? String(Device.modelName) : null;
    osVersion = Device.osVersion ? String(Device.osVersion) : null;
    deviceType = mapDeviceType(Device);
    const rawDeviceName = Device.deviceName ? String(Device.deviceName).trim() : "";
    // Ne jamais confondre avec le nom d'application
    if (
      rawDeviceName &&
      !isAppLabel(rawDeviceName) &&
      rawDeviceName !== (Application?.applicationName || "").trim()
    ) {
      deviceName = rawDeviceName;
    }
  }

  let appVersion: string | null = Application?.nativeApplicationVersion
    ? String(Application.nativeApplicationVersion)
    : null;
  const appBuild = Application?.nativeBuildVersion
    ? String(Application.nativeBuildVersion)
    : null;

  let expoRuntimeVersion: string | null = null;
  let otaUpdateId: string | null = null;
  let releaseChannel: string | null = null;
  let releaseSha: string | null = null;

  if (!appVersion) {
    try {
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      const Constants = require("expo-constants").default as {
        expoConfig?: { version?: string | null; extra?: Record<string, unknown> | null } | null;
      };
      appVersion = Constants?.expoConfig?.version
        ? String(Constants.expoConfig.version)
        : null;
      const extraSha = Constants?.expoConfig?.extra?.releaseSha;
      if (typeof extraSha === "string" && extraSha.trim()) {
        releaseSha = extraSha.trim().slice(0, 64);
      }
    } catch {
      /* noop */
    }
  }

  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const Updates = require("expo-updates") as {
      updateId?: string | null;
      runtimeVersion?: string | null;
      channel?: string | null;
      isEmbeddedLaunch?: boolean;
    };
    otaUpdateId = Updates?.updateId ? String(Updates.updateId) : "embedded";
    expoRuntimeVersion = Updates?.runtimeVersion
      ? String(Updates.runtimeVersion)
      : null;
    if (Updates?.channel) {
      releaseChannel = String(Updates.channel);
    } else if (Updates?.isEmbeddedLaunch) {
      releaseChannel = "embedded";
    }
  } catch {
    otaUpdateId = null;
  }

  return {
    platform: Platform.OS,
    deviceName,
    manufacturer,
    model,
    deviceType,
    osVersion,
    appVersion,
    appBuild,
    expoRuntimeVersion,
    otaUpdateId,
    releaseChannel,
    releaseSha,
  };
}

/**
 * Nom humain pour X-Device-Name : deviceName OS, sinon modèle, jamais le nom d'app.
 */
export function resolveDeviceHumanName(
  meta?: DeviceRuntimeMetadata | null
): string | null {
  const resolved = meta ?? resolveDeviceRuntimeMetadata();
  if (resolved.deviceName && !isAppLabel(resolved.deviceName)) {
    return resolved.deviceName.trim();
  }
  return null;
}

/**
 * Nom affichable prioritaire pour l'UI locale (modèle matériel).
 */
export function resolveDeviceDisplayName(
  meta?: DeviceRuntimeMetadata | null
): string {
  const resolved = meta ?? resolveDeviceRuntimeMetadata();
  const human = resolveDeviceHumanName(resolved);
  if (human) return human;
  if (resolved.model && resolved.model.trim()) {
    return resolved.model.trim();
  }
  if (resolved.manufacturer && resolved.manufacturer.trim()) {
    return resolved.manufacturer.trim();
  }
  if (resolved.platform === "ios") return "iPhone";
  if (resolved.platform === "android") return "Appareil Android";
  return "Appareil";
}

/**
 * Headers HTTP d'identité appareil (hors X-Device-ID, résolu séparément).
 * Compat N-1 : envoie aussi X-Platform (alias de X-Client-Platform).
 */
export function buildDeviceMetadataHeaders(
  meta?: DeviceRuntimeMetadata | null
): Record<string, string> {
  const resolved = meta ?? resolveDeviceRuntimeMetadata();
  const human = resolveDeviceHumanName(resolved);
  const headers: Record<string, string> = {
    "X-Client-Platform": resolved.platform,
    "X-Platform": resolved.platform,
    // X-Device-Name = nom humain OS si dispo, sinon modèle (jamais Lirie)
    "X-Device-Name": human || resolveDeviceDisplayName(resolved),
  };
  if (resolved.model) headers["X-Device-Model"] = resolved.model;
  if (resolved.manufacturer) headers["X-Device-Manufacturer"] = resolved.manufacturer;
  if (resolved.deviceType && resolved.deviceType !== "unknown") {
    headers["X-Device-Type"] = resolved.deviceType;
  }
  if (resolved.osVersion) headers["X-OS-Version"] = resolved.osVersion;
  if (resolved.appVersion) headers["X-App-Version"] = resolved.appVersion;
  if (resolved.appBuild) headers["X-App-Build"] = resolved.appBuild;
  return headers;
}
