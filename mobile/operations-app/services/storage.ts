// services/storage.ts
// Helper pour le stockage sécurisé et non-sécurisé des tokens et données d'authentification

import * as SecureStore from "expo-secure-store";
import { Platform, AppState, type AppStateStatus } from "react-native";
import { getLogger } from "@/utils/logger";
import { debugAuthLog, isDebugAuthEnabled } from "@/services/authDebug";
import { logAuthEvent } from "@/services/authLogging";
import { setAuthStateDegraded } from "@/services/authSync";

const log = getLogger("Storage");
import AsyncStorage from "@react-native-async-storage/async-storage";

// iOS Keychain : AFTER_FIRST_UNLOCK garantit l'accès aux tokens
// même quand l'app revient du background (le défaut WHEN_UNLOCKED peut
// échouer silencieusement après suspension).
const IOS_OPTS: SecureStore.SecureStoreOptions | undefined =
  Platform.OS === "ios"
    ? { keychainAccessible: SecureStore.AFTER_FIRST_UNLOCK }
    : undefined;

const secureGetRaw = (key: string) => SecureStore.getItemAsync(key, IOS_OPTS);
const secureSet = (key: string, value: string) =>
  SecureStore.setItemAsync(key, value, IOS_OPTS);
const secureDel = (key: string) => SecureStore.deleteItemAsync(key, IOS_OPTS);

/**
 * Wrapper autour de SecureStore.getItemAsync qui gere l'erreur iOS
 * "User interaction is not allowed" (Keychain inaccessible en background/locked).
 * Retourne null au lieu de throw pour eviter les crashes en production.
 */
const secureGet = async (key: string): Promise<string | null> => {
  try {
    return await secureGetRaw(key);
  } catch (error: any) {
    const msg = error?.message ?? String(error);
    if (msg.includes("User interaction is not allowed") || msg.includes("getValueWithKeyAsync")) {
      log.warn("keychain_locked", { key, error: msg });
      return null;
    }
    throw error;
  }
};

import * as Crypto from "expo-crypto";
import type { DriverAccountInfo } from "@/services/enterpriseDispatch";
import {
  DRIVER_AUTH_KEYS,
  ENTERPRISE_AUTH_KEYS,
  buildAuthNamespace,
} from "./storage/keys";

// ============ Clés de stockage sécurisé (SecureStore) ============
// Driver app: driver_* uniquement pour éviter mélange avec company dashboard (autre origine).
// Enterprise: enterprise.* pour le mode entreprise.
const SECURE_KEYS = {
  REFRESH_TOKEN: "driver_refresh_token",
  REFRESH_TOKEN_BACKUP: "driver_refresh_token_backup",
  ACCESS_TOKEN: "driver_access_token",
  USER_PUBLIC_ID: "driver_user_public_id",
  ENTERPRISE_TOKEN: "enterprise.token",
  ENTERPRISE_REFRESH: "enterprise.refresh",
  SAVED_EMAIL: "driver_saved_email",
  SAVED_PASSWORD: "driver_saved_password_encrypted",
} as const;

const ASYNC_NAMESPACE_KEYS = {
  DRIVER: "auth.namespace.driver",
  ENTERPRISE: "auth.namespace.enterprise",
} as const;

let activeDriverNamespaceCache: string | null = null;
let activeEnterpriseNamespaceCache: string | null = null;

async function getActiveNamespace(scope: "driver" | "enterprise"): Promise<string | null> {
  if (scope === "driver" && activeDriverNamespaceCache) return activeDriverNamespaceCache;
  if (scope === "enterprise" && activeEnterpriseNamespaceCache)
    return activeEnterpriseNamespaceCache;
  const key =
    scope === "driver" ? ASYNC_NAMESPACE_KEYS.DRIVER : ASYNC_NAMESPACE_KEYS.ENTERPRISE;
  const stored = await AsyncStorage.getItem(key);
  if (scope === "driver") activeDriverNamespaceCache = stored;
  else activeEnterpriseNamespaceCache = stored;
  return stored;
}

function withNamespace(baseKey: string, namespace: string | null): string {
  return namespace ? `${baseKey}:${namespace}` : baseKey;
}

async function setScopedSecureValue(
  scope: "driver" | "enterprise",
  baseKey: string,
  value: string
): Promise<void> {
  const namespace = await getActiveNamespace(scope);
  await secureSet(withNamespace(baseKey, namespace), value);
}

async function getScopedSecureValue(
  scope: "driver" | "enterprise",
  baseKey: string
): Promise<string | null> {
  const namespace = await getActiveNamespace(scope);
  const scopedKey = withNamespace(baseKey, namespace);
  let value = await secureGet(scopedKey);
  if (!value && namespace) {
    // Migration backward: fallback legacy puis write namespacé.
    const legacy = await secureGet(baseKey);
    if (legacy) {
      value = legacy;
      await secureSet(scopedKey, legacy).catch(() => {});
    }
  }
  return value;
}

async function removeScopedSecureValue(
  scope: "driver" | "enterprise",
  baseKey: string
): Promise<void> {
  const namespace = await getActiveNamespace(scope);
  await secureDel(withNamespace(baseKey, namespace));
  // Nettoyage legacy post-migration (best effort).
  if (namespace) {
    await secureDel(baseKey).catch(() => {});
  }
}

export async function setActiveAuthNamespace(params: {
  role: "driver" | "enterprise";
  userId: string | number;
  tenantId?: string | number | null;
  sessionId?: string | null;
}): Promise<string> {
  const namespace = buildAuthNamespace({
    role: params.role,
    userId: params.userId,
    tenantId: params.tenantId,
    sessionId: params.sessionId,
  });
  const key =
    params.role === "driver" ? ASYNC_NAMESPACE_KEYS.DRIVER : ASYNC_NAMESPACE_KEYS.ENTERPRISE;
  await AsyncStorage.setItem(key, namespace);
  if (params.role === "driver") activeDriverNamespaceCache = namespace;
  else activeEnterpriseNamespaceCache = namespace;
  return namespace;
}

// ============ Cache en mémoire pour optimisation des performances ============
// ⚡ Phase 1 : Cache en mémoire pour réduire les lectures SecureStore répétées
// Cache pour access_token (TTL: 1 minute - tokens expirent après 1h)
let cachedAccessToken: string | null = null;
let tokenCacheTime = 0;
const TOKEN_CACHE_TTL = 60000; // 1 minute

// Cache pour refresh_token (TTL: 5 minutes - moins fréquent)
let cachedRefreshToken: string | null = null;
let refreshTokenCacheTime = 0;
const REFRESH_TOKEN_CACHE_TTL = 300000; // 5 minutes

// ✅ CORRECTION : Cache pour Enterprise tokens (similaire à Driver)
// Cache pour enterprise access_token (TTL: 1 minute)
let cachedEnterpriseToken: string | null = null;
let enterpriseTokenCacheTime = 0;
const ENTERPRISE_TOKEN_CACHE_TTL = 60000; // 1 minute

// Cache pour enterprise refresh_token (TTL: 5 minutes)
let cachedEnterpriseRefreshToken: string | null = null;
let enterpriseRefreshTokenCacheTime = 0;
const ENTERPRISE_REFRESH_TOKEN_CACHE_TTL = 300000; // 5 minutes

// ⚡ Phase 4 : Métriques de performance (optionnel, pour debug)
// Uniquement actif en mode développement
let accessTokenCacheHitCount = 0;
let accessTokenCacheMissCount = 0;
let accessTokenTotalReadTime = 0;
let accessTokenReadCount = 0;

let refreshTokenCacheHitCount = 0;
let refreshTokenCacheMissCount = 0;
let refreshTokenTotalReadTime = 0;
let refreshTokenReadCount = 0;

const METRICS_LOG_INTERVAL = 100; // Log toutes les 100 lectures

// ============ Clés de stockage non-sécurisé (AsyncStorage) ============
const ASYNC_KEYS = {
  DRIVER_ID: "driver_id",
  DRIVER_ACCOUNT_INFO: "enterprise.driver_account_info", // Info du compte chauffeur associé
  // ✅ Device correlation (stable): utilisé pour X-Device-ID + push token multi-device
  // Note: clé historique, utilisée aussi côté enterprise
  DEVICE_ID: "enterprise.device_id",
  // Note : ACCESS_TOKEN a été déplacé vers SecureStore pour sécurité renforcée
} as const;

// ============ Stockage sécurisé (SecureStore) ============
export const secureStorage = {
  /**
   * Stocke le refresh token de manière sécurisée (Keychain/Keystore)
   * ⚡ Optimisation : Met à jour le cache en mémoire immédiatement
   */
  async setRefreshToken(token: string): Promise<void> {
    try {
      // Sauvegarder le backup et écrire le nouveau token en parallèle
      // pour réduire le temps de blocage du bridge natif (Android KeyStore)
      const currentPrimary =
        cachedRefreshToken ?? (await getScopedSecureValue("driver", SECURE_KEYS.REFRESH_TOKEN));
      const backupPromise = currentPrimary
        ? setScopedSecureValue("driver", SECURE_KEYS.REFRESH_TOKEN_BACKUP, currentPrimary).catch(() => {
            log.warn("refresh_backup_save_failed (non-blocking)");
          })
        : Promise.resolve();

      await Promise.all([
        backupPromise,
        setScopedSecureValue("driver", SECURE_KEYS.REFRESH_TOKEN, token),
      ]);

      cachedRefreshToken = token;
      refreshTokenCacheTime = Date.now();
    } catch (error) {
      log.error("refresh token save failed", { error });
      cachedRefreshToken = null;
      refreshTokenCacheTime = 0;
      throw error;
    }
  },

  /**
   * Récupère le refresh token depuis le stockage sécurisé
   * ⚡ Optimisation : Utilise le cache en mémoire si disponible et valide
   */
  async getRefreshToken(): Promise<string | null> {
    const startTime = __DEV__ ? Date.now() : 0;
    const now = Date.now();

    if (
      cachedRefreshToken &&
      now - refreshTokenCacheTime < REFRESH_TOKEN_CACHE_TTL
    ) {
      if (__DEV__) {
        refreshTokenCacheHitCount++;
      }
      return cachedRefreshToken;
    }

    if (__DEV__) {
      refreshTokenCacheMissCount++;
    }

    let token = await getScopedSecureValue("driver", SECURE_KEYS.REFRESH_TOKEN);

    // Fallback sur le backup si le primary est absent/corrompu
    if (!token) {
      const backup = await getScopedSecureValue("driver", SECURE_KEYS.REFRESH_TOKEN_BACKUP);
      if (backup) {
        log.warn("refresh_fallback_used: primary missing, using backup");
        token = backup;
        try {
          await setScopedSecureValue("driver", SECURE_KEYS.REFRESH_TOKEN, backup);
        } catch {
          // best-effort
        }
      }
    }

    cachedRefreshToken = token;
    refreshTokenCacheTime = now;

    if (__DEV__) {
      const readTime = Date.now() - startTime;
      refreshTokenTotalReadTime += readTime;
      refreshTokenReadCount++;

      if (refreshTokenReadCount % METRICS_LOG_INTERVAL === 0) {
        const avgReadTime =
          refreshTokenTotalReadTime / refreshTokenReadCount;
        const totalRequests =
          refreshTokenCacheHitCount + refreshTokenCacheMissCount;
        const cacheHitRate =
          totalRequests > 0
            ? (refreshTokenCacheHitCount / totalRequests) * 100
            : 0;
        log.info("refreshtoken performance", {
          avgMs: avgReadTime.toFixed(2),
          cacheHitRate: cacheHitRate.toFixed(1),
          hits: refreshTokenCacheHitCount,
          misses: refreshTokenCacheMissCount,
        });
      }
    }

    return token;
  },

  /**
   * Supprime le refresh token du stockage sécurisé
   * ⚡ Optimisation : Nettoie le cache en mémoire
   */
  async removeRefreshToken(): Promise<void> {
    await Promise.all([
      removeScopedSecureValue("driver", SECURE_KEYS.REFRESH_TOKEN),
      removeScopedSecureValue("driver", SECURE_KEYS.REFRESH_TOKEN_BACKUP),
    ]);

    cachedRefreshToken = null;
    refreshTokenCacheTime = 0;
  },

  /**
   * Stocke le public_id de l'utilisateur (pour auto-login)
   */
  async setUserPublicId(publicId: string): Promise<void> {
    await setScopedSecureValue("driver", SECURE_KEYS.USER_PUBLIC_ID, publicId);
  },

  async getUserPublicId(): Promise<string | null> {
    return await getScopedSecureValue("driver", SECURE_KEYS.USER_PUBLIC_ID);
  },

  async removeUserPublicId(): Promise<void> {
    await removeScopedSecureValue("driver", SECURE_KEYS.USER_PUBLIC_ID);
  },

  /**
   * Stocke le token d'accès de manière sécurisée (Keychain/Keystore)
   * ✅ Amélioration de sécurité : Même si court terme, le token d'accès est sensible
   * ⚡ Optimisation : Met à jour le cache en mémoire immédiatement
   */
  async setAccessToken(token: string): Promise<void> {
    try {
      await setScopedSecureValue("driver", SECURE_KEYS.ACCESS_TOKEN, token);

      cachedAccessToken = token;
      tokenCacheTime = Date.now();
    } catch (error) {
      log.error("access token save failed", { error });
      cachedAccessToken = null;
      tokenCacheTime = 0;
      throw error;
    }
  },

  /**
   * Récupère le token d'accès depuis le stockage sécurisé
   * ⚡ Optimisation : Utilise le cache en mémoire si disponible et valide
   * Réduit les lectures SecureStore répétées lors de requêtes simultanées
   */
  async getAccessToken(): Promise<string | null> {
    const startTime = __DEV__ ? Date.now() : 0;
    const now = Date.now();

    // Vérifier le cache en mémoire
    if (cachedAccessToken && now - tokenCacheTime < TOKEN_CACHE_TTL) {
      if (__DEV__) {
        accessTokenCacheHitCount++;
      }
      return cachedAccessToken;
    }

    if (__DEV__) {
      accessTokenCacheMissCount++;
    }

    const token = await getScopedSecureValue("driver", SECURE_KEYS.ACCESS_TOKEN);

    cachedAccessToken = token;
    tokenCacheTime = now;

    // ⚡ Phase 4 : Métriques de performance (dev uniquement)
    if (__DEV__) {
      const readTime = Date.now() - startTime;
      accessTokenTotalReadTime += readTime;
      accessTokenReadCount++;

      // Log périodique
      if (accessTokenReadCount % METRICS_LOG_INTERVAL === 0) {
        const avgReadTime = accessTokenTotalReadTime / accessTokenReadCount;
        const totalRequests =
          accessTokenCacheHitCount + accessTokenCacheMissCount;
        const cacheHitRate =
          totalRequests > 0
            ? (accessTokenCacheHitCount / totalRequests) * 100
            : 0;
        log.info("accesstoken performance", {
          avgMs: avgReadTime.toFixed(2),
          cacheHitRate: cacheHitRate.toFixed(1),
          hits: accessTokenCacheHitCount,
          misses: accessTokenCacheMissCount,
        });
      }
    }

    return token;
  },

  /**
   * Supprime le token d'accès du stockage sécurisé
   * ⚡ Optimisation : Nettoie le cache en mémoire
   */
  async removeAccessToken(): Promise<void> {
    await removeScopedSecureValue("driver", SECURE_KEYS.ACCESS_TOKEN);

    cachedAccessToken = null;
    tokenCacheTime = 0;
  },

  /**
   * P1.A — Nettoie uniquement les clés auth chauffeur (SecureStore + AsyncStorage).
   * Ne touche pas aux tokens Enterprise, device_id, remember me, etc.
   */
  async clearDriverAuthOnly(): Promise<void> {
    await Promise.all([
      ...DRIVER_AUTH_KEYS.secure.map((k) => secureDel(k)),
      AsyncStorage.multiRemove([...DRIVER_AUTH_KEYS.async]),
    ]);

    cachedAccessToken = null;
    tokenCacheTime = 0;
    cachedRefreshToken = null;
    refreshTokenCacheTime = 0;

    if (__DEV__) {
      accessTokenCacheHitCount = 0;
      accessTokenCacheMissCount = 0;
      accessTokenTotalReadTime = 0;
      accessTokenReadCount = 0;
      refreshTokenCacheHitCount = 0;
      refreshTokenCacheMissCount = 0;
      refreshTokenTotalReadTime = 0;
      refreshTokenReadCount = 0;
    }
  },

  /**
   * P1.A — Nettoie uniquement les clés auth entreprise (SecureStore + AsyncStorage).
   * Ne touche pas aux tokens chauffeur, device_id, etc.
   */
  async clearEnterpriseAuthOnly(): Promise<void> {
    await Promise.all([
      ...ENTERPRISE_AUTH_KEYS.secure.map((k) => secureDel(k)),
      AsyncStorage.multiRemove([...ENTERPRISE_AUTH_KEYS.async]),
    ]);

    cachedEnterpriseToken = null;
    enterpriseTokenCacheTime = 0;
    cachedEnterpriseRefreshToken = null;
    enterpriseRefreshTokenCacheTime = 0;
  },

  /**
   * Nettoie tout le stockage auth (driver + enterprise).
   * ⚠️ Réservé aux tests / dev / factory reset. En prod : no-op pour éviter effacement accidentel.
   */
  async clearAll(): Promise<void> {
    if (!__DEV__) {
      log.warn("clearall blocked in production", {});
      return;
    }
    await Promise.all([
      secureDel(SECURE_KEYS.REFRESH_TOKEN),
      secureDel(SECURE_KEYS.REFRESH_TOKEN_BACKUP),
      secureDel(SECURE_KEYS.ACCESS_TOKEN),
      secureDel(SECURE_KEYS.USER_PUBLIC_ID),
      secureDel(SECURE_KEYS.ENTERPRISE_TOKEN),
      secureDel(SECURE_KEYS.ENTERPRISE_REFRESH),
    ]);

    cachedAccessToken = null;
    tokenCacheTime = 0;
    cachedRefreshToken = null;
    refreshTokenCacheTime = 0;
    cachedEnterpriseToken = null;
    enterpriseTokenCacheTime = 0;
    cachedEnterpriseRefreshToken = null;
    enterpriseRefreshTokenCacheTime = 0;

    if (__DEV__) {
      accessTokenCacheHitCount = 0;
      accessTokenCacheMissCount = 0;
      accessTokenTotalReadTime = 0;
      accessTokenReadCount = 0;
      refreshTokenCacheHitCount = 0;
      refreshTokenCacheMissCount = 0;
      refreshTokenTotalReadTime = 0;
      refreshTokenReadCount = 0;
    }
  },

  // ============ PHASE 1 : Mémorisation des identifiants ============
  /**
   * ✅ PHASE 1 : Sauvegarde les identifiants (email + mot de passe chiffré)
   * Le mot de passe est chiffré avec SHA-256 + salt (email) avant stockage
   * ⚠️ IMPORTANT : Stocke le mot de passe en clair mais dans SecureStore (Keychain/Keystore)
   * Le SecureStore est déjà sécurisé par le système d'exploitation
   * @param email Email de l'utilisateur
   * @param password Mot de passe en clair (sera stocké dans SecureStore sécurisé)
   */
  async saveCredentials(email: string, password: string): Promise<void> {
    try {
      // Stocker l'email et le mot de passe dans SecureStore
      // SecureStore utilise Keychain (iOS) / Keystore (Android) qui sont sécurisés
      await Promise.all([
        secureSet(SECURE_KEYS.SAVED_EMAIL, email),
        secureSet(SECURE_KEYS.SAVED_PASSWORD, password),
      ]);
      log.success("credentials saved for auto-login", {});
    } catch (error) {
      log.error("credentials save failed", { error });
      throw error;
    }
  },

  /**
   * ✅ PHASE 1 : Récupère les identifiants sauvegardés
   * @returns { email: string | null, password: string | null }
   */
  async getSavedCredentials(): Promise<{
    email: string | null;
    password: string | null;
  }> {
    try {
      const [email, password] = await Promise.all([
        secureGet(SECURE_KEYS.SAVED_EMAIL),
        secureGet(SECURE_KEYS.SAVED_PASSWORD),
      ]);
      return { email, password };
    } catch (error) {
      log.error("credentials get failed", { error });
      return { email: null, password: null };
    }
  },

  /**
   * ✅ PHASE 1 : Supprime les identifiants sauvegardés
   */
  async clearSavedCredentials(): Promise<void> {
    try {
      await Promise.all([
        secureDel(SECURE_KEYS.SAVED_EMAIL),
        secureDel(SECURE_KEYS.SAVED_PASSWORD),
      ]);
      log.success("saved credentials cleared", {});
    } catch (error) {
      log.error("credentials clear failed", { error });
    }
  },

  // ============ Stockage Enterprise (SecureStore avec cache) ============
  /**
   * ✅ CORRECTION : Stocke le token Enterprise de manière sécurisée
   * ⚡ Optimisation : Met à jour le cache en mémoire immédiatement
   */
  async setEnterpriseToken(token: string): Promise<void> {
    try {
      await setScopedSecureValue("enterprise", SECURE_KEYS.ENTERPRISE_TOKEN, token);

      cachedEnterpriseToken = token;
      enterpriseTokenCacheTime = Date.now();
    } catch (error) {
      log.error("enterprise token save failed", { error });
      cachedEnterpriseToken = null;
      enterpriseTokenCacheTime = 0;
      throw error;
    }
  },

  /**
   * ✅ CORRECTION : Récupère le token Enterprise depuis SecureStore
   * ⚡ Optimisation : Utilise le cache en mémoire si disponible et valide
   */
  async getEnterpriseToken(): Promise<string | null> {
    const now = Date.now();

    // Vérifier le cache en mémoire
    if (
      cachedEnterpriseToken &&
      now - enterpriseTokenCacheTime < ENTERPRISE_TOKEN_CACHE_TTL
    ) {
      return cachedEnterpriseToken;
    }

    const token = await getScopedSecureValue("enterprise", SECURE_KEYS.ENTERPRISE_TOKEN);

    cachedEnterpriseToken = token;
    enterpriseTokenCacheTime = now;

    return token;
  },

  /**
   * ✅ CORRECTION : Supprime le token Enterprise du stockage sécurisé
   * ⚡ Optimisation : Nettoie le cache en mémoire
   */
  async removeEnterpriseToken(): Promise<void> {
    await removeScopedSecureValue("enterprise", SECURE_KEYS.ENTERPRISE_TOKEN);

    cachedEnterpriseToken = null;
    enterpriseTokenCacheTime = 0;
  },

  /**
   * ✅ CORRECTION : Stocke le refresh token Enterprise de manière sécurisée
   * ⚡ Optimisation : Met à jour le cache en mémoire immédiatement
   */
  async setEnterpriseRefreshToken(token: string): Promise<void> {
    try {
      await setScopedSecureValue("enterprise", SECURE_KEYS.ENTERPRISE_REFRESH, token);

      cachedEnterpriseRefreshToken = token;
      enterpriseRefreshTokenCacheTime = Date.now();
      if (isDebugAuthEnabled()) {
        debugAuthLog("ent_refresh_write", {
          key: SECURE_KEYS.ENTERPRISE_REFRESH,
          len: token.length,
        });
      }
    } catch (error) {
      log.error("enterprise refresh token save failed", { error });
      cachedEnterpriseRefreshToken = null;
      enterpriseRefreshTokenCacheTime = 0;
      throw error; // Propager l'erreur pour que l'app puisse réagir
    }
  },

  /**
   * ✅ CORRECTION : Récupère le refresh token Enterprise depuis SecureStore
   * ⚡ Optimisation : Utilise le cache en mémoire si disponible et valide
   */
  async getEnterpriseRefreshToken(): Promise<string | null> {
    const now = Date.now();

    // Vérifier le cache en mémoire
    if (
      cachedEnterpriseRefreshToken &&
      now - enterpriseRefreshTokenCacheTime < ENTERPRISE_REFRESH_TOKEN_CACHE_TTL
    ) {
      return cachedEnterpriseRefreshToken;
    }

    const token = await getScopedSecureValue("enterprise", SECURE_KEYS.ENTERPRISE_REFRESH);

    cachedEnterpriseRefreshToken = token;
    enterpriseRefreshTokenCacheTime = now;

    if (isDebugAuthEnabled()) {
      debugAuthLog("ent_refresh_read", {
        key: SECURE_KEYS.ENTERPRISE_REFRESH,
        present: token ? 1 : 0,
        len: token?.length ?? 0,
      });
    }
    return token;
  },

  /**
   * ✅ CORRECTION : Supprime le refresh token Enterprise du stockage sécurisé
   * ⚡ Optimisation : Nettoie le cache en mémoire
   */
  async removeEnterpriseRefreshToken(): Promise<void> {
    await removeScopedSecureValue("enterprise", SECURE_KEYS.ENTERPRISE_REFRESH);

    cachedEnterpriseRefreshToken = null;
    enterpriseRefreshTokenCacheTime = 0;
  },

  /**
   * ✅ CORRECTION : Nettoie tout le stockage Enterprise sécurisé
   * ⚡ Optimisation : Nettoie également tous les caches en mémoire
   */
  async clearEnterpriseTokens(): Promise<void> {
    await Promise.all([
      removeScopedSecureValue("enterprise", SECURE_KEYS.ENTERPRISE_TOKEN),
      removeScopedSecureValue("enterprise", SECURE_KEYS.ENTERPRISE_REFRESH),
    ]);


    cachedEnterpriseToken = null;
    enterpriseTokenCacheTime = 0;
    cachedEnterpriseRefreshToken = null;
    enterpriseRefreshTokenCacheTime = 0;
  },

  /**
   * ⚡ Phase 4 : Récupère les métriques de performance (dev uniquement)
   * Utile pour le debugging et l'analyse des performances
   */
  getPerformanceMetrics() {
    if (!__DEV__) {
      return null;
    }

    const accessTokenTotal =
      accessTokenCacheHitCount + accessTokenCacheMissCount;
    const refreshTokenTotal =
      refreshTokenCacheHitCount + refreshTokenCacheMissCount;

    return {
      accessToken: {
        cacheHits: accessTokenCacheHitCount,
        cacheMisses: accessTokenCacheMissCount,
        totalRequests: accessTokenTotal,
        cacheHitRate:
          accessTokenTotal > 0
            ? (accessTokenCacheHitCount / accessTokenTotal) * 100
            : 0,
        avgReadTime:
          accessTokenReadCount > 0
            ? accessTokenTotalReadTime / accessTokenReadCount
            : 0,
        totalReads: accessTokenReadCount,
      },
      refreshToken: {
        cacheHits: refreshTokenCacheHitCount,
        cacheMisses: refreshTokenCacheMissCount,
        totalRequests: refreshTokenTotal,
        cacheHitRate:
          refreshTokenTotal > 0
            ? (refreshTokenCacheHitCount / refreshTokenTotal) * 100
            : 0,
        avgReadTime:
          refreshTokenReadCount > 0
            ? refreshTokenTotalReadTime / refreshTokenReadCount
            : 0,
        totalReads: refreshTokenReadCount,
      },
    };
  },
};

export type CommitSessionTokensParams = {
  scope: "driver" | "enterprise";
  accessToken: string;
  refreshToken?: string | null;
  expiresAt?: number | null;
  sessionMeta?: Record<string, unknown>;
  sessionStorageKey?: string;
  trigger_source?: string;
};

type SessionCommitEvent = {
  scope: "driver" | "enterprise";
  trigger_source: string;
  success: boolean;
  has_refresh: boolean;
};

const sessionCommitListeners = new Set<(event: SessionCommitEvent) => void>();

export function addSessionCommitListener(
  listener: (event: SessionCommitEvent) => void
): () => void {
  sessionCommitListeners.add(listener);
  return () => sessionCommitListeners.delete(listener);
}

function emitSessionCommitEvent(event: SessionCommitEvent): void {
  sessionCommitListeners.forEach((listener) => {
    try {
      listener(event);
    } catch (error) {
      log.warn("session commit listener failed", {
        error: error instanceof Error ? error.message : String(error),
      });
    }
  });
}

function decodeJwtPayload(token: string): Record<string, any> | null {
  try {
    return JSON.parse(atob(token.split(".")[1]));
  } catch {
    return null;
  }
}

/**
 * Commit ordonné des tokens de session pour éviter les désynchronisations
 * mémoire/stockage lors d'une rotation de token.
 */
export async function commitSessionTokensAtomically(
  params: CommitSessionTokensParams
): Promise<void> {
  const now = Date.now();
  try {
    if (params.scope === "driver") {
      const payload = decodeJwtPayload(params.accessToken);
      if (payload?.sub || payload?.user_id || payload?.public_id) {
        await setActiveAuthNamespace({
          role: "driver",
          userId: payload?.public_id || payload?.user_id || payload?.sub,
          tenantId: payload?.company_id ?? null,
          sessionId: payload?.session_id ?? null,
        });
      }
    } else if (params.sessionMeta) {
      const meta = params.sessionMeta as any;
      await setActiveAuthNamespace({
        role: "enterprise",
        userId: meta?.user?.public_id || meta?.user?.id || "unknown",
        tenantId: meta?.company?.id ?? null,
        sessionId: meta?.sessionId ?? meta?.session_id ?? null,
      });
    }

    if (params.scope === "driver") {
      await secureStorage.setAccessToken(params.accessToken);
      if (params.refreshToken) {
        await secureStorage.setRefreshToken(params.refreshToken);
      }
    } else {
      await secureStorage.setEnterpriseToken(params.accessToken);
      if (params.refreshToken) {
        await secureStorage.setEnterpriseRefreshToken(params.refreshToken);
      }
    }

    if (params.sessionStorageKey) {
      const payload = {
        ...(params.sessionMeta ?? {}),
        expiresAt: params.expiresAt ?? null,
        updatedAt: now,
      };
      await AsyncStorage.setItem(params.sessionStorageKey, JSON.stringify(payload));
    }

    logAuthEvent("AUTH_COMMIT_SUCCESS", {
      route: params.scope,
      trigger_source: params.trigger_source || "unknown",
      has_refresh: Boolean(params.refreshToken),
    });
    emitSessionCommitEvent({
      scope: params.scope,
      trigger_source: params.trigger_source || "unknown",
      success: true,
      has_refresh: Boolean(params.refreshToken),
    });
  } catch (error) {
    setAuthStateDegraded();
    logAuthEvent("AUTH_COMMIT_PARTIAL_FAILURE", {
      route: params.scope,
      trigger_source: params.trigger_source || "unknown",
      has_refresh: Boolean(params.refreshToken),
    });
    emitSessionCommitEvent({
      scope: params.scope,
      trigger_source: params.trigger_source || "unknown",
      success: false,
      has_refresh: Boolean(params.refreshToken),
    });
    throw error;
  }
}

// ============ Stockage non-sécurisé (AsyncStorage) ============
export const asyncStorage = {
  /**
   * Récupère l'identifiant stable de l'appareil (si présent)
   */
  async getDeviceId(): Promise<string | null> {
    return await AsyncStorage.getItem(ASYNC_KEYS.DEVICE_ID);
  },

  /**
   * Génère (si besoin) et retourne un identifiant stable d'appareil.
   * Utilisé pour corréler refresh tokens / sessions / push tokens multi-device.
   */
  async getOrCreateDeviceId(): Promise<string> {
    let stored = await AsyncStorage.getItem(ASYNC_KEYS.DEVICE_ID);
    if (!stored) {
      if (typeof Crypto.randomUUID === "function") {
        stored = Crypto.randomUUID();
      } else {
        const bytes = await Crypto.getRandomBytesAsync(16);
        const bytesArray = Array.from(bytes as Uint8Array);
        stored = bytesArray
          .map((byte) => byte.toString(16).padStart(2, "0"))
          .join("");
      }
      await AsyncStorage.setItem(ASYNC_KEYS.DEVICE_ID, stored);
    }
    if (!stored) {
      throw new Error("Impossible de générer un identifiant appareil");
    }
    return stored;
  },

  /**
   * Stocke l'ID du chauffeur (pour navigation rapide)
   * Note : L'access_token a été déplacé vers SecureStore pour plus de sécurité
   */
  async setDriverId(driverId: number): Promise<void> {
    await AsyncStorage.setItem(ASYNC_KEYS.DRIVER_ID, String(driverId));
  },

  /**
   * Récupère l'ID du chauffeur
   */
  async getDriverId(): Promise<number | null> {
    const id = await AsyncStorage.getItem(ASYNC_KEYS.DRIVER_ID);
    return id ? parseInt(id, 10) : null;
  },

  /**
   * Supprime l'ID du chauffeur
   */
  async removeDriverId(): Promise<void> {
    await AsyncStorage.removeItem(ASYNC_KEYS.DRIVER_ID);
  },

  /**
   * Stocke l'info du compte chauffeur associé (pour éviter de refaire l'appel API)
   */
  async setDriverAccountInfo(info: DriverAccountInfo): Promise<void> {
    await AsyncStorage.setItem(
      ASYNC_KEYS.DRIVER_ACCOUNT_INFO,
      JSON.stringify(info)
    );
  },

  /**
   * Récupère l'info du compte chauffeur associé
   */
  async getDriverAccountInfo(): Promise<DriverAccountInfo | null> {
    const info = await AsyncStorage.getItem(ASYNC_KEYS.DRIVER_ACCOUNT_INFO);
    return info ? (JSON.parse(info) as DriverAccountInfo) : null;
  },

  /**
   * Supprime l'info du compte chauffeur associé
   */
  async removeDriverAccountInfo(): Promise<void> {
    await AsyncStorage.removeItem(ASYNC_KEYS.DRIVER_ACCOUNT_INFO);
  },

  /**
   * Nettoie tout le stockage d'authentification non-sécurisé (IDs uniquement)
   * Note : Les tokens sont maintenant gérés par secureStorage
   */
  async clearAuth(): Promise<void> {
    await AsyncStorage.multiRemove([
      ASYNC_KEYS.DRIVER_ID,
      ASYNC_KEYS.DRIVER_ACCOUNT_INFO,
    ]);
  },
};

// ============ Migration iOS Keychain accessibility ============
// Les items ecrits avant l'ajout de AFTER_FIRST_UNLOCK ont l'accessibilite
// par defaut WHEN_UNLOCKED → illisibles en background/locked → crash
// "User interaction is not allowed". Cette migration les re-ecrit une seule fois.
const KEYCHAIN_MIGRATION_KEY = "@atmr:keychain_migrated_v1";

async function migrateKeychainAccessibility(): Promise<void> {
  if (Platform.OS !== "ios") return;
  try {
    const done = await AsyncStorage.getItem(KEYCHAIN_MIGRATION_KEY);
    if (done) return;

    const allKeys = Object.values(SECURE_KEYS);
    for (const key of allKeys) {
      try {
        const value = await secureGetRaw(key);
        if (value) {
          await secureSet(key, value);
        }
      } catch {
        // L'item est peut-etre deja inaccessible — on skip
      }
    }

    await AsyncStorage.setItem(KEYCHAIN_MIGRATION_KEY, "1");
    log.info("keychain migration done", { keys: allKeys.length });
  } catch (e) {
    log.warn("keychain migration failed", { error: e });
  }
}

// Lancer la migration quand l'app est active (foreground)
if (Platform.OS === "ios") {
  const runMigrationWhenActive = () => {
    const handleState = (state: AppStateStatus) => {
      if (state === "active") {
        migrateKeychainAccessibility();
        sub.remove();
      }
    };
    const sub = AppState.addEventListener("change", handleState);
    // Si deja active, lancer immediatement
    if (AppState.currentState === "active") {
      migrateKeychainAccessibility();
      sub.remove();
    }
  };
  runMigrationWhenActive();
}

