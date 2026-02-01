/**
 * P2.2 — Contexte pour logs SRE-grade (multi-tenant, corrélation).
 *
 * Alimenté par les infos déjà en mémoire (driver/enterprise session).
 * Jamais de PII brute — user_public_id et device_id sont hashés.
 */

import { Platform } from "react-native";
import { asyncStorage } from "./storage";
import Constants from "expo-constants";

export type LogContextSnapshot = {
  company_id?: number;
  user_public_id_hash?: string;
  device_id_hash?: string;
  platform: "ios" | "android" | "web";
  build_number?: string;
  app_version?: string;
  release_channel?: string;
  git_sha?: string;
};

let contextCache: LogContextSnapshot = {
  platform: Platform.OS as "ios" | "android" | "web",
  build_number: Constants.expoConfig?.ios?.buildNumber ?? Constants.expoConfig?.android?.versionCode?.toString(),
  app_version: Constants.expoConfig?.version ?? "?",
  release_channel: Constants.expoConfig?.extra?.releaseChannel as string | undefined,
  git_sha: process.env.EXPO_PUBLIC_GIT_SHA as string | undefined,
};

/** Hash async (sha256 + 12 chars / 48 bits) — fire-and-forget, met à jour le cache. Réduit collisions multi-tenant. */
async function hashForLog(value: string | undefined | null): Promise<string | undefined> {
  if (!value || typeof value !== "string") return undefined;
  try {
    const Crypto = require("expo-crypto").default ?? require("expo-crypto");
    const alg = Crypto.CryptoDigestAlgorithm?.SHA256 ?? "SHA-256";
    const digest = await Crypto.digestStringAsync(alg, value);
    return digest.slice(0, 12);
  } catch {
    return undefined;
  }
}

/**
 * Met à jour le contexte utilisateur (company_id, user_public_id, device_id).
 * Hash async — le cache est mis à jour quand le hash est prêt.
 */
export function setLogContextUser(params: {
  company_id?: number;
  user_public_id?: string | null;
  device_id?: string | null;
}): void {
  if (params.company_id != null) {
    contextCache = { ...contextCache, company_id: params.company_id };
  }
  if (params.user_public_id) {
    hashForLog(params.user_public_id).then((h) => {
      if (h) contextCache = { ...contextCache, user_public_id_hash: h };
    });
  }
  if (params.device_id) {
    hashForLog(params.device_id).then((h) => {
      if (h) contextCache = { ...contextCache, device_id_hash: h };
    });
  }
}

/**
 * Retourne le snapshot actuel (sync).
 */
export function getLogContextSnapshot(): LogContextSnapshot {
  return { ...contextCache };
}

/**
 * Init device_id pour logs (fire-and-forget).
 * À appeler au boot (ex: _layout).
 */
export function initLogContext(): void {
  asyncStorage.getOrCreateDeviceId().then((id) => {
    setLogContextUser({ device_id: id });
  }).catch(() => {});
}
