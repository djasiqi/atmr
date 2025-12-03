// services/storage.ts
// Helper pour le stockage sécurisé et non-sécurisé des tokens et données d'authentification

import * as SecureStore from "expo-secure-store";
import AsyncStorage from "@react-native-async-storage/async-storage";

// ============ Clés de stockage sécurisé (SecureStore) ============
const SECURE_KEYS = {
  REFRESH_TOKEN: "auth.refresh_token",
  ACCESS_TOKEN: "auth.access_token", // ✅ Amélioration : Access token aussi dans SecureStore
  USER_PUBLIC_ID: "auth.user_public_id", // Optionnel, pour auto-login
} as const;

// ============ Clés de stockage non-sécurisé (AsyncStorage) ============
const ASYNC_KEYS = {
  DRIVER_ID: "driver_id",
  // Note : ACCESS_TOKEN a été déplacé vers SecureStore pour sécurité renforcée
} as const;

// ============ Stockage sécurisé (SecureStore) ============
export const secureStorage = {
  /**
   * Stocke le refresh token de manière sécurisée (Keychain/Keystore)
   */
  async setRefreshToken(token: string): Promise<void> {
    await SecureStore.setItemAsync(SECURE_KEYS.REFRESH_TOKEN, token);
  },

  /**
   * Récupère le refresh token depuis le stockage sécurisé
   */
  async getRefreshToken(): Promise<string | null> {
    return await SecureStore.getItemAsync(SECURE_KEYS.REFRESH_TOKEN);
  },

  /**
   * Supprime le refresh token du stockage sécurisé
   */
  async removeRefreshToken(): Promise<void> {
    await SecureStore.deleteItemAsync(SECURE_KEYS.REFRESH_TOKEN);
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
   */
  async setAccessToken(token: string): Promise<void> {
    await SecureStore.setItemAsync(SECURE_KEYS.ACCESS_TOKEN, token);
  },

  /**
   * Récupère le token d'accès depuis le stockage sécurisé
   */
  async getAccessToken(): Promise<string | null> {
    return await SecureStore.getItemAsync(SECURE_KEYS.ACCESS_TOKEN);
  },

  /**
   * Supprime le token d'accès du stockage sécurisé
   */
  async removeAccessToken(): Promise<void> {
    await SecureStore.deleteItemAsync(SECURE_KEYS.ACCESS_TOKEN);
  },

  /**
   * Nettoie tout le stockage sécurisé (refresh_token, access_token, user_public_id)
   */
  async clearAll(): Promise<void> {
    await Promise.all([
      SecureStore.deleteItemAsync(SECURE_KEYS.REFRESH_TOKEN),
      SecureStore.deleteItemAsync(SECURE_KEYS.ACCESS_TOKEN),
      SecureStore.deleteItemAsync(SECURE_KEYS.USER_PUBLIC_ID),
    ]);
  },
};

// ============ Stockage non-sécurisé (AsyncStorage) ============
export const asyncStorage = {
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
   * Nettoie tout le stockage d'authentification non-sécurisé (IDs uniquement)
   * Note : Les tokens sont maintenant gérés par secureStorage
   */
  async clearAuth(): Promise<void> {
    await AsyncStorage.multiRemove([ASYNC_KEYS.DRIVER_ID]);
  },
};

