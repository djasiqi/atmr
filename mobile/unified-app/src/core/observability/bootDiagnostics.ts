import { PixelRatio, Platform } from "react-native";
import * as Application from "expo-application";
import * as Device from "expo-device";
import * as Sentry from "@sentry/react-native";
import * as Updates from "expo-updates";
import { appendSessionJournalEvent } from "./sessionJournal";

export type BootFallbackName =
  | "BootSplashFallbackTriggered"
  | "LandingRevealFallbackTriggered"
  | "LandingRevealHardTimeout"
  | "ProfileRevealFallbackTriggered"
  | "NotificationChannelsMissing";

/** Detection runtime de la New Architecture (Fabric / Bridgeless), sans dependre d'app.json. */
function detectNewArchEnabled(): boolean {
  const g = globalThis as Record<string, unknown>;
  return g.RN$Bridgeless === true || g.nativeFabricUIManager != null;
}

/** Garde-fou anti-spam: 1 evenement par type de fallback et par session (process). */
const reportedThisSession = new Set<BootFallbackName>();

export function reportBootFallback(
  name: BootFallbackName,
  extra?: Record<string, unknown>,
): void {
  if (reportedThisSession.has(name)) {
    return;
  }
  reportedThisSession.add(name);

  try {
    const tags = {
      device_model: Device.modelName ?? "unknown",
      android_api_level: Platform.OS === "android" ? String(Platform.Version) : "n/a",
      os_version: Device.osVersion ?? "unknown",
      app_version: Application.nativeApplicationVersion ?? "unknown",
      ota_update_id: Updates.updateId ?? "embedded",
      new_arch_enabled: String(detectNewArchEnabled()),
      font_scale: String(PixelRatio.getFontScale()),
    };
    // `fingerprint` explicite : ces événements sont émis depuis un setTimeout, donc
    // la stack synthétique (Hermes) est mal mappée vers des symboles arbitraires
    // (ex. FlatList.props.renderItem). Sans fingerprint, Sentry grouperait par cette
    // stack trompeuse → issues mal nommées/attribuées et fragmentées par appareil.
    // On regroupe donc par nom d'événement.
    Sentry.captureMessage(name, {
      level: "warning",
      tags,
      extra: extra ?? {},
      fingerprint: [name],
    });
  } catch {
    // monitoring ne doit pas casser le boot
  }

  void appendSessionJournalEvent(`boot.fallback.${name}`, {
    ...extra,
    device_model: Device.modelName ?? null,
    android_api_level: Platform.OS === "android" ? Platform.Version : null,
    new_arch_enabled: detectNewArchEnabled(),
  });
}
