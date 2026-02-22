// services/biometricAuth.ts
// ✅ PHASE 2 : Service d'authentification biométrique pour reconnexion rapide

import * as LocalAuthentication from "expo-local-authentication";
import { getLogger } from "@/utils/logger";
import { loginDriver } from "@/services/api";

const log = getLogger("Biometric");
import { secureStorage } from "@/services/storage";
import {
  getRememberedCredentials,
  setRememberMe,
  clearRememberedCredentials,
} from "@/utils/rememberMeStorage";

/** Lancée quand aucun identifiant mémorisé alors que la biométrie est proposée → éviter boucle UX "biométrie échoue sans raison". */
export class BiometricNoCredentialsError extends Error {
  constructor() {
    super("Identifiants mémorisés indisponibles.");
    this.name = "BiometricNoCredentialsError";
  }
}

/**
 * ✅ PHASE 2 : Vérifie si l'authentification biométrique est disponible sur l'appareil
 * @returns true si l'appareil supporte la biométrie ET qu'un identifiant biométrique est enregistré
 */
export async function isBiometricAvailable(): Promise<boolean> {
  try {
    const compatible = await LocalAuthentication.hasHardwareAsync();
    if (!compatible) {
      log.info("device not compatible with biometrics", {});
      return false;
    }

    const enrolled = await LocalAuthentication.isEnrolledAsync();
    if (!enrolled) {
      log.info("no biometric enrolled on device", {});
      return false;
    }

    log.success("biometric auth available", {});
    return true;
  } catch (error) {
    log.error("biometric availability check failed", { error });
    return false;
  }
}

/**
 * ✅ PHASE 2 : Récupère les types d'authentification biométrique disponibles
 * @returns Liste des types disponibles (ex: ["fingerprint", "facial"])
 */
export async function getAvailableBiometricTypes(): Promise<
  LocalAuthentication.AuthenticationType[]
> {
  try {
    const types = await LocalAuthentication.supportedAuthenticationTypesAsync();
    log.info("available biometric types", { types });
    return types;
  } catch (error) {
    log.error("get biometric types failed", { error });
    return [];
  }
}

/**
 * ✅ PHASE 2 : Authentifie l'utilisateur avec la biométrie (empreinte digitale, Face ID, etc.)
 * @param options Options d'authentification (message, labels, etc.)
 * @returns true si l'authentification a réussi, false sinon
 */
export async function authenticateWithBiometric(
  options?: {
    promptMessage?: string;
    cancelLabel?: string;
    disableDeviceFallback?: boolean;
    fallbackLabel?: string;
  }
): Promise<boolean> {
  try {
    const result = await LocalAuthentication.authenticateAsync({
      promptMessage: options?.promptMessage || "Authentification requise",
      cancelLabel: options?.cancelLabel || "Annuler",
      disableDeviceFallback: options?.disableDeviceFallback ?? false, // Permet code PIN si biométrie échoue
      fallbackLabel: options?.fallbackLabel || "Utiliser le code PIN",
    });

    if (result.success) {
      log.success("biometric auth succeeded", {});
      return true;
    } else {
      log.info("biometric auth cancelled or failed", { error: result.error });
      return false;
    }
  } catch (error) {
    log.error("biometric auth failed", { error });
    return false;
  }
}

/**
 * ✅ PHASE 2 : Auto-login avec authentification biométrique
 * Règle : la biométrie dépend de "Se souvenir de moi" (identifiants mémorisés dans rememberMeStorage).
 * useAuth n'appelle cette fonction que lorsqu'aucun token valide n'est disponible (refresh échoué ou absent),
 * donc on n'exige jamais le mot de passe si un token est disponible.
 * Récupère les identifiants mémorisés, demande la biométrie, puis login si succès.
 * @param options Options d'authentification biométrique
 * @returns true si l'auto-login a réussi, false sinon
 */
export async function autoLoginWithBiometric(
  options?: {
    promptMessage?: string;
    cancelLabel?: string;
    disableDeviceFallback?: boolean;
    fallbackLabel?: string;
  }
): Promise<boolean> {
  try {
    // 1. Vérifier si la biométrie est disponible
    const available = await isBiometricAvailable();
    if (!available) {
      log.info("biometrics not available, auto-login skipped", {});
      return false;
    }

    // 2. Récupérer les identifiants mémorisés (SecureStore, "Se souvenir de moi")
    const savedCreds = await getRememberedCredentials();
    if (!savedCreds?.email || !savedCreds?.password) {
      await setRememberMe(false);
      await clearRememberedCredentials();
      throw new BiometricNoCredentialsError();
    }

    // 3. Demander l'authentification biométrique
    const authenticated = await authenticateWithBiometric({
      promptMessage: options?.promptMessage || "Authentifiez-vous pour vous reconnecter",
      cancelLabel: options?.cancelLabel || "Annuler",
      disableDeviceFallback: options?.disableDeviceFallback ?? false,
      fallbackLabel: options?.fallbackLabel || "Utiliser le code PIN",
    });

    if (!authenticated) {
      log.info("biometric auth cancelled by user", {});
      return false;
    }

    log.info("attempting login with saved credentials", {});

    const loginResponse = await loginDriver(savedCreds.email, savedCreds.password);

    // 5. Stocker les nouveaux tokens
    if (loginResponse.token) {
      await secureStorage.setAccessToken(loginResponse.token);
    }
    if (loginResponse.refresh_token) {
      await secureStorage.setRefreshToken(loginResponse.refresh_token);
    }
    if (loginResponse.user?.public_id) {
      await secureStorage.setUserPublicId(loginResponse.user.public_id);
    }

    log.success("auto-login with biometric succeeded", {});
    return true;
  } catch (error: any) {
    if (error instanceof BiometricNoCredentialsError) throw error;
    log.error("auto-login with biometric failed", { error });
    return false;
  }
}
