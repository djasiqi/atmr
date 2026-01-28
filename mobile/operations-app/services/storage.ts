// services/storage.ts
// Helper pour le stockage sécurisé et non-sécurisé des tokens et données d'authentification

import * as SecureStore from "expo-secure-store";
import { debugAuthLog, isDebugAuthEnabled } from "@/services/authDebug";
import AsyncStorage from "@react-native-async-storage/async-storage";
import * as Crypto from "expo-crypto";
import type { DriverAccountInfo } from "@/services/enterpriseDispatch";

// ============ Clés de stockage sécurisé (SecureStore) ============
// Namespaces : auth.* = tokens chauffeur ; enterprise.* = tokens entreprise ; driver.* = rememberMe (voir rememberMeStorage)
const SECURE_KEYS = {
  REFRESH_TOKEN: "auth.refresh_token",
  ACCESS_TOKEN: "auth.access_token",
  USER_PUBLIC_ID: "auth.user_public_id",
  ENTERPRISE_TOKEN: "enterprise.token",
  ENTERPRISE_REFRESH: "enterprise.refresh",
  // Legacy (préférez driver.* via rememberMeStorage pour le flux chauffeur)
  SAVED_EMAIL: "auth.saved_email",
  SAVED_PASSWORD: "auth.saved_password_encrypted",
} as const;

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
      await SecureStore.setItemAsync(SECURE_KEYS.REFRESH_TOKEN, token);

      // Mettre à jour le cache immédiatement
      cachedRefreshToken = token;
      refreshTokenCacheTime = Date.now();
    } catch (error) {
      // ✅ CORRECTION : Gérer les erreurs de stockage (ex: Keychain/Keystore inaccessible)
      console.error("[Storage] ❌ Erreur lors de la sauvegarde du refresh token:", error);
      // Ne pas mettre à jour le cache si le stockage a échoué
      cachedRefreshToken = null;
      refreshTokenCacheTime = 0;
      throw error; // Propager l'erreur pour que l'app puisse réagir
    }
  },

  /**
   * Récupère le refresh token depuis le stockage sécurisé
   * ⚡ Optimisation : Utilise le cache en mémoire si disponible et valide
   */
  async getRefreshToken(): Promise<string | null> {
    const startTime = __DEV__ ? Date.now() : 0;
    const now = Date.now();

    // Vérifier le cache en mémoire
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

    // Lire depuis SecureStore
    const token = await SecureStore.getItemAsync(SECURE_KEYS.REFRESH_TOKEN);

    // Mettre à jour le cache
    cachedRefreshToken = token;
    refreshTokenCacheTime = now;

    // ⚡ Phase 4 : Métriques de performance (dev uniquement)
    if (__DEV__) {
      const readTime = Date.now() - startTime;
      refreshTokenTotalReadTime += readTime;
      refreshTokenReadCount++;

      // Log périodique
      if (refreshTokenReadCount % METRICS_LOG_INTERVAL === 0) {
        const avgReadTime =
          refreshTokenTotalReadTime / refreshTokenReadCount;
        const totalRequests =
          refreshTokenCacheHitCount + refreshTokenCacheMissCount;
        const cacheHitRate =
          totalRequests > 0
            ? (refreshTokenCacheHitCount / totalRequests) * 100
            : 0;
        console.log(
          `[Storage] RefreshToken Performance: avg=${avgReadTime.toFixed(2)}ms, cache=${cacheHitRate.toFixed(1)}%, hits=${refreshTokenCacheHitCount}, misses=${refreshTokenCacheMissCount}`
        );
      }
    }

    return token;
  },

  /**
   * Supprime le refresh token du stockage sécurisé
   * ⚡ Optimisation : Nettoie le cache en mémoire
   */
  async removeRefreshToken(): Promise<void> {
    await SecureStore.deleteItemAsync(SECURE_KEYS.REFRESH_TOKEN);

    // Nettoyer le cache
    cachedRefreshToken = null;
    refreshTokenCacheTime = 0;
  },

  /**
   * Stocke le public_id de l'utilisateur (pour auto-login)
   */
  async setUserPublicId(publicId: string): Promise<void> {
    await SecureStore.setItemAsync(SECURE_KEYS.USER_PUBLIC_ID, publicId);
  },

  /**
   * Récupère le public_id de l'utilisateur
   */
  async getUserPublicId(): Promise<string | null> {
    return await SecureStore.getItemAsync(SECURE_KEYS.USER_PUBLIC_ID);
  },

  /**
   * Supprime le public_id de l'utilisateur
   */
  async removeUserPublicId(): Promise<void> {
    await SecureStore.deleteItemAsync(SECURE_KEYS.USER_PUBLIC_ID);
  },

  /**
   * Stocke le token d'accès de manière sécurisée (Keychain/Keystore)
   * ✅ Amélioration de sécurité : Même si court terme, le token d'accès est sensible
   * ⚡ Optimisation : Met à jour le cache en mémoire immédiatement
   */
  async setAccessToken(token: string): Promise<void> {
    try {
      await SecureStore.setItemAsync(SECURE_KEYS.ACCESS_TOKEN, token);

      // Mettre à jour le cache immédiatement
      cachedAccessToken = token;
      tokenCacheTime = Date.now();
    } catch (error) {
      // ✅ CORRECTION : Gérer les erreurs de stockage (ex: Keychain/Keystore inaccessible)
      console.error("[Storage] ❌ Erreur lors de la sauvegarde du access token:", error);
      // Ne pas mettre à jour le cache si le stockage a échoué
      cachedAccessToken = null;
      tokenCacheTime = 0;
      throw error; // Propager l'erreur pour que l'app puisse réagir
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

    // Lire depuis SecureStore
    const token = await SecureStore.getItemAsync(SECURE_KEYS.ACCESS_TOKEN);

    // Mettre à jour le cache
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
        console.log(
          `[Storage] AccessToken Performance: avg=${avgReadTime.toFixed(2)}ms, cache=${cacheHitRate.toFixed(1)}%, hits=${accessTokenCacheHitCount}, misses=${accessTokenCacheMissCount}`
        );
      }
    }

    return token;
  },

  /**
   * Supprime le token d'accès du stockage sécurisé
   * ⚡ Optimisation : Nettoie le cache en mémoire
   */
  async removeAccessToken(): Promise<void> {
    await SecureStore.deleteItemAsync(SECURE_KEYS.ACCESS_TOKEN);

    // Nettoyer le cache
    cachedAccessToken = null;
    tokenCacheTime = 0;
  },

  /**
   * Nettoie tout le stockage sécurisé (refresh_token, access_token, user_public_id)
   * ⚡ Optimisation : Nettoie également tous les caches en mémoire
   */
  async clearAll(): Promise<void> {
    await Promise.all([
      SecureStore.deleteItemAsync(SECURE_KEYS.REFRESH_TOKEN),
      SecureStore.deleteItemAsync(SECURE_KEYS.ACCESS_TOKEN),
      SecureStore.deleteItemAsync(SECURE_KEYS.USER_PUBLIC_ID),
    ]);

    // Nettoyer tous les caches en mémoire
    cachedAccessToken = null;
    tokenCacheTime = 0;
    cachedRefreshToken = null;
    refreshTokenCacheTime = 0;

    // ⚡ Phase 4 : Réinitialiser les métriques
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
        SecureStore.setItemAsync(SECURE_KEYS.SAVED_EMAIL, email),
        SecureStore.setItemAsync(SECURE_KEYS.SAVED_PASSWORD, password),
      ]);
      if (__DEV__) {
        console.log("[Storage] ✅ Identifiants sauvegardés pour auto-login");
      }
    } catch (error) {
      console.error("[Storage] ❌ Erreur lors de la sauvegarde des identifiants:", error);
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
        SecureStore.getItemAsync(SECURE_KEYS.SAVED_EMAIL),
        SecureStore.getItemAsync(SECURE_KEYS.SAVED_PASSWORD),
      ]);
      return { email, password };
    } catch (error) {
      console.error("[Storage] ❌ Erreur lors de la récupération des identifiants:", error);
      return { email: null, password: null };
    }
  },

  /**
   * ✅ PHASE 1 : Supprime les identifiants sauvegardés
   */
  async clearSavedCredentials(): Promise<void> {
    try {
      await Promise.all([
        SecureStore.deleteItemAsync(SECURE_KEYS.SAVED_EMAIL),
        SecureStore.deleteItemAsync(SECURE_KEYS.SAVED_PASSWORD),
      ]);
      if (__DEV__) {
        console.log("[Storage] ✅ Identifiants sauvegardés supprimés");
      }
    } catch (error) {
      console.error("[Storage] ❌ Erreur lors de la suppression des identifiants:", error);
    }
  },

  // ============ Stockage Enterprise (SecureStore avec cache) ============
  /**
   * ✅ CORRECTION : Stocke le token Enterprise de manière sécurisée
   * ⚡ Optimisation : Met à jour le cache en mémoire immédiatement
   */
  async setEnterpriseToken(token: string): Promise<void> {
    try {
      await SecureStore.setItemAsync(SECURE_KEYS.ENTERPRISE_TOKEN, token);

      // Mettre à jour le cache immédiatement
      cachedEnterpriseToken = token;
      enterpriseTokenCacheTime = Date.now();
    } catch (error) {
      // ✅ CORRECTION : Gérer les erreurs de stockage (ex: Keychain/Keystore inaccessible)
      console.error("[Storage] ❌ Erreur lors de la sauvegarde du token Enterprise:", error);
      // Ne pas mettre à jour le cache si le stockage a échoué
      cachedEnterpriseToken = null;
      enterpriseTokenCacheTime = 0;
      throw error; // Propager l'erreur pour que l'app puisse réagir
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

    // Lire depuis SecureStore
    const token = await SecureStore.getItemAsync(SECURE_KEYS.ENTERPRISE_TOKEN);

    // Mettre à jour le cache
    cachedEnterpriseToken = token;
    enterpriseTokenCacheTime = now;

    return token;
  },

  /**
   * ✅ CORRECTION : Supprime le token Enterprise du stockage sécurisé
   * ⚡ Optimisation : Nettoie le cache en mémoire
   */
  async removeEnterpriseToken(): Promise<void> {
    await SecureStore.deleteItemAsync(SECURE_KEYS.ENTERPRISE_TOKEN);

    // Nettoyer le cache
    cachedEnterpriseToken = null;
    enterpriseTokenCacheTime = 0;
  },

  /**
   * ✅ CORRECTION : Stocke le refresh token Enterprise de manière sécurisée
   * ⚡ Optimisation : Met à jour le cache en mémoire immédiatement
   */
  async setEnterpriseRefreshToken(token: string): Promise<void> {
    try {
      await SecureStore.setItemAsync(SECURE_KEYS.ENTERPRISE_REFRESH, token);

      // Mettre à jour le cache immédiatement
      cachedEnterpriseRefreshToken = token;
      enterpriseRefreshTokenCacheTime = Date.now();
      if (isDebugAuthEnabled()) {
        debugAuthLog("ent_refresh_write", {
          key: SECURE_KEYS.ENTERPRISE_REFRESH,
          len: token.length,
        });
      }
    } catch (error) {
      // ✅ CORRECTION : Gérer les erreurs de stockage (ex: Keychain/Keystore inaccessible)
      console.error("[Storage] ❌ Erreur lors de la sauvegarde du refresh token Enterprise:", error);
      // Ne pas mettre à jour le cache si le stockage a échoué
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

    // Lire depuis SecureStore
    const token = await SecureStore.getItemAsync(
      SECURE_KEYS.ENTERPRISE_REFRESH
    );

    // Mettre à jour le cache
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
    await SecureStore.deleteItemAsync(SECURE_KEYS.ENTERPRISE_REFRESH);

    // Nettoyer le cache
    cachedEnterpriseRefreshToken = null;
    enterpriseRefreshTokenCacheTime = 0;
  },

  /**
   * ✅ CORRECTION : Nettoie tout le stockage Enterprise sécurisé
   * ⚡ Optimisation : Nettoie également tous les caches en mémoire
   */
  async clearEnterpriseTokens(): Promise<void> {
    await Promise.all([
      SecureStore.deleteItemAsync(SECURE_KEYS.ENTERPRISE_TOKEN),
      SecureStore.deleteItemAsync(SECURE_KEYS.ENTERPRISE_REFRESH),
    ]);

    // Nettoyer tous les caches en mémoire
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

