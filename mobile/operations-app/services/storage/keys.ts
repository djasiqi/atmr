/**
 * P1.A — Listes canoniques des clés d'authentification
 *
 * Source unique de vérité pour les clés à supprimer lors d'un logout.
 * Garantit qu'aucune clé auth n'est oubliée et qu'aucune clé non-auth n'est supprimée.
 *
 * @see storage.ts — clearDriverAuthOnly(), clearEnterpriseAuthOnly()
 *
 * ## Migration (P1.A)
 * - Avant : clearAll() supprimait tout le SecureStore driver ; clearAuth() supprimait driver_id + driver_account_info.
 * - Après : clearDriverAuthOnly() / clearEnterpriseAuthOnly() suppriment uniquement les clés listées ici.
 * - Clés legacy non migrées : aucune (les clés actuelles correspondent aux listes).
 * - clearAll() reste disponible pour tests/dev/factory reset ; warn en prod si appelé.
 */

// ============ Driver Auth ============
/** Clés SecureStore utilisées pour l'authentification chauffeur */
export const DRIVER_AUTH_SECURE_KEYS = [
  "driver_refresh_token",
  "driver_refresh_token_backup",
  "driver_access_token",
  "driver_user_public_id",
] as const;

/** Clés AsyncStorage utilisées pour l'authentification chauffeur */
export const DRIVER_AUTH_ASYNC_KEYS = [
  "driver_id",
  "enterprise.driver_account_info", // Info compte chauffeur (namespace "enterprise." = app operations)
] as const;

/** Toutes les clés auth chauffeur (SecureStore + AsyncStorage) */
export const DRIVER_AUTH_KEYS = {
  secure: DRIVER_AUTH_SECURE_KEYS,
  async: DRIVER_AUTH_ASYNC_KEYS,
} as const;

// ============ Enterprise Auth ============
/** Clés SecureStore utilisées pour l'authentification entreprise */
export const ENTERPRISE_AUTH_SECURE_KEYS = [
  "enterprise.token",
  "enterprise.refresh",
] as const;

/** Clés AsyncStorage utilisées pour l'authentification entreprise */
export const ENTERPRISE_AUTH_ASYNC_KEYS = [
  "enterprise.session",
  "enterprise_session_just_created",
] as const;

/** Toutes les clés auth entreprise (SecureStore + AsyncStorage) */
export const ENTERPRISE_AUTH_KEYS = {
  secure: ENTERPRISE_AUTH_SECURE_KEYS,
  async: ENTERPRISE_AUTH_ASYNC_KEYS,
} as const;

// ============ Non-auth (documentation) ============
/**
 * Clés qui ne doivent PAS être supprimées lors d'un logout auth.
 * Exemples : device_id, auth.mode, préférences UI, caches fonctionnels.
 */
export const NON_AUTH_KEYS = [
  "enterprise.device_id",
  "auth.mode",
  "driver_saved_email",
  "driver_saved_password_encrypted",
] as const;
