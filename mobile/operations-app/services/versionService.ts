// services/versionService.ts
// Service pour vérifier la version de l'application et gérer les mises à jour

import { Platform } from "react-native";
import * as Application from "expo-application";
import Constants from "expo-constants";
import { api } from "./api";

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
    console.warn("expo-application non disponible, fallback...", e);
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
    console.warn("Constants.expoConfig non disponible, fallback...", e);
  }

  // Dernier recours: package.json (nécessite un require)
  try {
    const pkg = require("../package.json");
    if (pkg?.version) {
      return pkg.version;
    }
  } catch (e) {
    console.warn("package.json non accessible", e);
  }

  // Valeur par défaut si rien ne fonctionne
  console.warn("Impossible de récupérer la version, utilisation de '1.0.0'");
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
  const platform = getCurrentPlatform();
  const currentVersion = getCurrentAppVersion();

  try {
    const response = await api.post<VersionCheckResponse>("/app/version-check", {
      platform,
      current_version: currentVersion,
    });

    return response.data;
  } catch (error: any) {
    // En cas d'erreur réseau ou serveur, on considère que tout est OK
    // pour ne pas bloquer l'utilisateur
    console.warn("Erreur lors de la vérification de version:", error);

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
  }
}

