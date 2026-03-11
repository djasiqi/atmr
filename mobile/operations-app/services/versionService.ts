// services/versionService.ts
// Service pour vérifier la version de l'application et gérer les mises à jour

import { Platform } from "react-native";
import * as Application from "expo-application";
import Constants from "expo-constants";
import { getLogger } from "@/utils/logger";
import { api } from "./api";

const log = getLogger("Version");
let versionCheckInFlight: Promise<VersionCheckResponse> | null = null;

export type UpdateStatus = "OK" | "UPDATE_RECOMMENDED" | "UPDATE_REQUIRED";

export interface VersionCheckRequest {
  platform: "android" | "ios";
  current_version: string;
}

export interface VersionCheckResponse {
  platform: "android" | "ios";
  current_version: string;
  latest_version: string;
  min_required_version: string;
  status: UpdateStatus;
  store_url: string | null;
  message: string | null;
}

function shouldBlockOnUpdateRequired(): boolean {
  const appVariant = String(
    Constants.expoConfig?.extra?.APP_VARIANT || process.env.APP_VARIANT || "prod"
  );
  const explicitFlag = process.env.EXPO_PUBLIC_ENABLE_FORCE_UPDATE;
  if (explicitFlag === "true") return true;
  if (explicitFlag === "false") return false;
  // Par défaut: on ne bloque pas les builds review/prod sans flag explicite.
  return appVariant !== "prod";
}

function normalizeVersionStatus(
  response: VersionCheckResponse
): VersionCheckResponse {
  if (response.status !== "UPDATE_REQUIRED") {
    return response;
  }
  if (shouldBlockOnUpdateRequired()) {
    return response;
  }
  log.warn("update required downgraded for review safety", {
    current_version: response.current_version,
    latest_version: response.latest_version,
    min_required_version: response.min_required_version,
  });
  return {
    ...response,
    status: "UPDATE_RECOMMENDED",
    message:
      response.message ||
      "Une mise à jour est disponible. Veuillez mettre à jour dès que possible.",
  };
}

/**
 * Récupère la version actuelle de l'application.
 * 
 * Priorité:
 * 1. expo-application (recommandé pour Expo)
 * 2. Constants.expoConfig (fallback)
 * 3. package.json version (dernier recours)
 */
export function getCurrentAppVersion(): string {
  try {
    // Essayer expo-application en premier (recommandé pour Expo)
    if (Application.nativeApplicationVersion) {
      return Application.nativeApplicationVersion;
    }
  } catch (e) {
    log.warn("expo-application not available, fallback", { error: e });
  }

  try {
    // Fallback: Constants.expoConfig
    const configVersion =
      Constants.expoConfig?.ios?.version ||
      Constants.expoConfig?.android?.version;
    if (configVersion) {
      return configVersion;
    }
  } catch (e) {
    log.warn("constants expo config not available, fallback", { error: e });
  }

  // Dernier recours: package.json (nécessite un require)
  try {
    const pkg = require("../package.json");
    if (pkg?.version) {
      return pkg.version;
    }
  } catch (e) {
    log.warn("package.json not accessible", { error: e });
  }

  log.warn("could not get version, using 1.0.0", {});
  return "1.0.0";
}

/**
 * Récupère la plateforme actuelle (android ou ios).
 */
export function getCurrentPlatform(): "android" | "ios" {
  return Platform.OS === "android" ? "android" : "ios";
}

/**
 * Vérifie la version de l'application auprès du backend.
 * 
 * @returns Promise<VersionCheckResponse> - Réponse du backend avec le statut de mise à jour
 * @throws Error si la requête échoue
 */
export async function checkVersion(): Promise<VersionCheckResponse> {
  if (versionCheckInFlight) {
    return versionCheckInFlight;
  }

  const platform = getCurrentPlatform();
  const currentVersion = getCurrentAppVersion();

  versionCheckInFlight = (async () => {
    try {
      const response = await api.post<VersionCheckResponse>("/app/version-check", {
        platform,
        current_version: currentVersion,
      });

      return normalizeVersionStatus(response.data);
    } catch (error: any) {
      log.warn("version check failed", { error });

      // Retourner une réponse par défaut "OK" pour ne pas bloquer l'app
      return {
        platform,
        current_version: currentVersion,
        latest_version: currentVersion,
        min_required_version: currentVersion,
        status: "OK",
        store_url: null,
        message: null,
      };
    } finally {
      versionCheckInFlight = null;
    }
  })();

  return versionCheckInFlight;
}

