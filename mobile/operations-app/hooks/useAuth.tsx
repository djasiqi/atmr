import React, {
  createContext,
  ReactNode,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { Alert, AppState, AppStateStatus, InteractionManager } from "react-native";
import AsyncStorage from "@react-native-async-storage/async-storage";
import * as Crypto from "expo-crypto";

import {
  api,
  AuthResponse,
  Driver,
  fetchDriverProfile,
  loginDriver,
  invalidateInterceptorCache,
} from "@/services/api";
import {
  secureStorage,
  asyncStorage,
  setActiveAuthNamespace,
} from "@/services/storage";
import {
  getRememberMe,
  setRememberMe as persistRememberMe,
  setRememberedCredentials,
  clearRememberedCredentials,
  RememberMeStorageError,
} from "@/utils/rememberMeStorage";
// ✅ CORRECTION : Les tokens Enterprise sont maintenant dans SecureStore
import {
  ENTERPRISE_SESSION_KEY,
  EnterpriseLoginParams,
  EnterpriseLoginResponse,
  EnterpriseLoginMfaPayload,
  EnterpriseTokenPayload,
  fetchEnterpriseSession,
  loginEnterprise,
  refreshEnterpriseToken,
  runEnterpriseRefreshSingleflight,
  verifyEnterpriseMfa,
  invalidateEnterpriseInterceptorCache,
} from "@/services/enterpriseAuth";
import {
  type AuthSessionState,
  assertSessionPurgeAllowed,
  getAuthBootstrapState,
  notifyAuthReady,
  notifyAuthNotReady,
  subscribeAuthSessionState,
} from "@/services/authSync";
import { setLogContextUser } from "@/services/logContext";
import {
  type ForceLogoutMetadata,
  invokeForceLogoutDriver,
  invokeForceLogoutEnterprise,
  registerForceLogoutDriver,
  registerForceLogoutEnterprise,
  type DriverLogoutReason,
  type EnterpriseLogoutReason,
} from "@/services/authController";
import {
  getAuthFailureReason,
  isAuthNotReadyError,
  shouldLogoutFromRefreshFailure,
} from "@/services/authGuards";
import { isNetworkError, isHttpAuthError, getHttpStatus } from "@/utils/authErrorHelpers";
import {
  isDriverProactiveRefreshInCooldown,
  getDriverProactiveRefreshCooldownRemaining,
  recordDriverProactiveRefreshFailure,
  resetDriverProactiveRefreshCooldown,
  isEnterpriseProactiveRefreshInCooldown,
  getEnterpriseProactiveRefreshCooldownRemaining,
  recordEnterpriseProactiveRefreshFailure,
  resetEnterpriseProactiveRefreshCooldown,
} from "@/services/proactiveRefreshCooldown";
import { debugAuthLog, isDebugAuthEnabled } from "@/services/authDebug";
import { pushSessionEvent } from "@/services/sessionJournal";
import { setLogoutMarker, shouldShowLogoutBanner } from "@/services/logoutMarker";
import { connectSocket, disconnectSocket } from "@/services/socket";
import { ensureBackgroundTrackingStopped } from "@/services/locationTracker";
// ✅ PHASE 2 : Import de l'authentification biométrique
import {
  autoLoginWithBiometric,
  BiometricNoCredentialsError,
} from "@/services/biometricAuth";
import { sendIngestEvent } from "@/src/config/telemetry";
import { getLogger } from "@/utils/logger";
import { refreshDriverTokenOrchestrated } from "@/services/driverTokenOrchestrator";
import { buildAuthNamespace } from "@/services/storage/keys";

const log = getLogger("Auth");

const debugLog = (data: Record<string, unknown>) => {
  if (__DEV__) {
    try {
      sendIngestEvent(data);
    } catch {
      // ignore
    }
  }
};

const MODE_KEY = "auth.mode";
const ENTERPRISE_DEVICE_KEY = "enterprise.device_id";
/**
 * Verrou global de bootstrap auth (single-run par instance).
 * Reset au unmount pour que StrictMode remount re-exécute le bootstrap
 * et restaure l'état (mode, driver) depuis le storage.
 */
let authBootstrapOncePromise: Promise<void> | null = null;

type AuthMode = "driver" | "enterprise";

/**
 * Décoder un JWT et extraire le timestamp d'expiration (exp)
 * @returns timestamp d'expiration en millisecondes, ou null si décodage échoue
 */
const getTokenExpiration = (token: string): number | null => {
  try {
    const payload = JSON.parse(atob(token.split(".")[1]));
    return payload.exp ? payload.exp * 1000 : null; // Convertir en ms
  } catch (error) {
    log.warn("jwt decode failed", { error });
    return null;
  }
};

/** P0.2.A — Access token encore valide (avec skew pour horloges). P0.3.B : skew 2min pour devices mal réglés. */
const accessStillValid = (token: string | null, skewMs = 2 * 60 * 1000): boolean => {
  if (!token) return false;
  const exp = getTokenExpiration(token);
  if (!exp) return false;
  return Date.now() < exp - skewMs;
};

const buildEnterpriseSessionKeyFromState = (
  session: EnterpriseSessionState | null
): string => {
  if (!session) {
    return buildAuthNamespace({
      role: "enterprise",
      userId: "unknown",
      tenantId: null,
      sessionId: null,
    });
  }
  return buildAuthNamespace({
    role: "enterprise",
    userId: session.user?.public_id || "unknown",
    tenantId: session.company?.id ?? null,
    sessionId: session.sessionId ?? null,
  });
};

interface EnterpriseSessionState {
  token: string;
  refreshToken: string | null;
  user: EnterpriseTokenPayload["user"];
  company: {
    id: number;
    name: string;
    dispatchMode?: string | null;
  };
  scopes: string[];
  sessionId: string;
}

/**
 * Cache d'état auth au niveau module — survit aux remounts (StrictMode, hot reload).
 * Évite le flash enterprise + arrêt sync/socket quand AuthProvider remonte avec état initial.
 */
let authStateCache: {
  mode: AuthMode;
  driver: Driver | null;
  driverToken: string | null;
  enterpriseSession: EnterpriseSessionState | null;
  initialLoading: boolean;
} | null = null;

interface EnterpriseMfaChallenge {
  challengeId: string;
  ttl?: number;
  methods: string[];
  message?: string;
}

interface AuthContextType {
  mode: AuthMode;
  setMode: (mode: AuthMode) => Promise<void>;
  switchMode: (mode: AuthMode) => Promise<void>;
  loading: boolean;
  deviceId: string | null;

  driver: Driver | null;
  token: string | null;
  isDriverAuthenticated: boolean;
  driverLoading: boolean;
  login: (email: string, password: string, rememberMe?: boolean) => Promise<void>;
  logout: () => Promise<void>;
  refreshProfile: () => Promise<void>;

  enterpriseSession: EnterpriseSessionState | null;
  isEnterpriseAuthenticated: boolean;
  enterpriseLoading: boolean;
  pendingEnterpriseMfa: EnterpriseMfaChallenge | null;
  loginEnterprise: (
    params: EnterpriseLoginParams & { rememberMe?: boolean }
  ) => Promise<
    | { mfaRequired: true; challenge: EnterpriseMfaChallenge }
    | { mfaRequired: false }
  >;
  verifyEnterpriseMfa: (code: string, challengeId?: string) => Promise<void>;
  refreshEnterprise: () => Promise<void>;
  loadEnterpriseSession: () => Promise<void>;
  loadDriverSession: () => Promise<void>;
  logoutEnterprise: () => Promise<void>;

  isAuthenticated: boolean;
  authSessionState: AuthSessionState;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

const parseEnterpriseSuccess = (
  payload: EnterpriseTokenPayload
): EnterpriseSessionState => ({
  token: payload.token,
  refreshToken: payload.refresh_token ?? null,
  user: payload.user,
  company: {
    id: payload.company.id,
    name: payload.company.name,
    dispatchMode:
      (payload.company as any).dispatchMode ?? payload.company.dispatch_mode,
  },
  scopes: payload.scopes ?? [],
  sessionId: payload.session_id,
});

export const AuthProvider = ({ children }: { children: ReactNode }) => {
  const [mode, setModeState] = useState<AuthMode>(
    () => authStateCache?.mode ?? "enterprise"
  );
  const [initialLoading, setInitialLoading] = useState(
    () => authStateCache?.initialLoading ?? true
  );
  const [deviceId, setDeviceId] = useState<string | null>(null);

  const [driver, setDriver] = useState<Driver | null>(
    () => authStateCache?.driver ?? null
  );
  const [driverToken, setDriverToken] = useState<string | null>(
    () => authStateCache?.driverToken ?? null
  );
  const [driverLoading, setDriverLoading] = useState(false);

  const [enterpriseSession, setEnterpriseSession] =
    useState<EnterpriseSessionState | null>(
      () => authStateCache?.enterpriseSession ?? null
    );
  const [enterpriseLoading, setEnterpriseLoading] = useState(false);
  const [pendingEnterpriseMfa, setPendingEnterpriseMfa] =
    useState<EnterpriseMfaChallenge | null>(null);
  const [pendingEnterpriseRememberMe, setPendingEnterpriseRememberMe] = useState<{
    email: string;
    password: string;
  } | null>(null);
  const [authSessionState, setAuthSessionState] = useState<AuthSessionState>(
    () => getAuthBootstrapState()
  );

  /** Sync état → cache module (survit remount StrictMode / hot reload). */
  useEffect(() => {
    authStateCache = {
      mode,
      driver,
      driverToken,
      enterpriseSession,
      initialLoading,
    };
  }, [mode, driver, driverToken, enterpriseSession, initialLoading]);

  useEffect(() => {
    setAuthSessionState(getAuthBootstrapState());
    return subscribeAuthSessionState((state) => {
      setAuthSessionState(state);
    });
  }, []);

  /** P0.1 — Garde anti double-exécution (driver). */
  const driverLogoutInProgressRef = useRef(false);
  /** P0.1 — Garde anti double-exécution (enterprise). */
  const enterpriseLogoutInProgressRef = useRef(false);
  /** P0.3.A — Timeout refresh proactif driver (initial + retry). */
  const driverProactiveRefreshTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  /** P0.3.A — Timeout refresh proactif enterprise (initial + retry). */
  const enterpriseProactiveRefreshTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const storeMode = useCallback(async (nextMode: AuthMode) => {
    setModeState(nextMode);
    await AsyncStorage.setItem(MODE_KEY, nextMode);
  }, []);

  const ensureDeviceId = useCallback(async (): Promise<string> => {
    if (deviceId) return deviceId;
    let stored = await AsyncStorage.getItem(ENTERPRISE_DEVICE_KEY);
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
      await AsyncStorage.setItem(ENTERPRISE_DEVICE_KEY, stored);
    }
    if (!stored) {
      throw new Error("Impossible de générer un identifiant appareil");
    }
    setDeviceId(stored);
    return stored;
  }, [deviceId]);

  /** P1.A — Nettoie uniquement les clés auth chauffeur (SecureStore + AsyncStorage). */
  const clearDriverStorage = useCallback(async () => {
    await secureStorage.clearDriverAuthOnly();
  }, []);

  /** P1.A — Nettoie uniquement les clés auth entreprise (SecureStore + AsyncStorage). */
  const clearEnterpriseStorage = useCallback(async () => {
    await secureStorage.clearEnterpriseAuthOnly();
  }, []);

  /** P0 — Porte de sortie unique pour invalider la session driver (storage + état + socket). */
  const forceLogoutDriverInternal = useCallback(
    async (reason: DriverLogoutReason, metadata?: ForceLogoutMetadata) => {
      if (driverLogoutInProgressRef.current) return;
      driverLogoutInProgressRef.current = true;
      setDriverLoading(true);
      notifyAuthNotReady();
      try {
        if (!metadata?.severity || !metadata?.trigger_source || !metadata?.source) {
          log.warn("forceLogout driver metadata missing in callback", { reason });
        }
        assertSessionPurgeAllowed(reason);
        await ensureBackgroundTrackingStopped("logout");
        if (shouldShowLogoutBanner(reason)) {
          await setLogoutMarker({ route: "driver", reason, ts: Date.now() });
        }
        if (reason === "manual_logout") {
          const keepRememberedCreds = await getRememberMe();
          const accessToken = await secureStorage.getAccessToken();
          const refreshToken = await secureStorage.getRefreshToken();
          if (accessToken) {
            try {
              await api.post(
                "/auth/logout",
                { refresh_token: refreshToken ?? null },
                { headers: { Authorization: `Bearer ${accessToken}` } }
              );
            } catch (e) {
              log.warn("server-side logout error", { error: e });
            }
          }
          await secureStorage.clearDriverAuthOnly();
          if (!keepRememberedCreds) {
            await clearRememberedCredentials();
          }
        } else {
          await secureStorage.clearDriverAuthOnly();
        }
        invalidateInterceptorCache();
        disconnectSocket();
        setDriver(null);
        setDriverToken(null);
      } finally {
        driverLogoutInProgressRef.current = false;
        setDriverLoading(false);
      }
    },
    [getRememberMe, clearRememberedCredentials]
  );

  /** P0 — Porte de sortie unique pour invalider la session enterprise (storage + état). */
  const forceLogoutEnterpriseInternal = useCallback(
    async (reason: EnterpriseLogoutReason, metadata?: ForceLogoutMetadata) => {
      if (enterpriseLogoutInProgressRef.current) return;
      enterpriseLogoutInProgressRef.current = true;
      try {
        if (!metadata?.severity || !metadata?.trigger_source || !metadata?.source) {
          log.warn("forceLogout enterprise metadata missing in callback", { reason });
        }
        notifyAuthNotReady();
        assertSessionPurgeAllowed(reason);
        if (shouldShowLogoutBanner(reason)) {
          await setLogoutMarker({ route: "enterprise", reason, ts: Date.now() });
        }
        await clearEnterpriseStorage();
        const keepRememberedCreds = await getRememberMe("enterprise");
        if (!keepRememberedCreds) {
          await clearRememberedCredentials("enterprise");
        }
        invalidateEnterpriseInterceptorCache();
        setEnterpriseSession(null);
        setPendingEnterpriseMfa(null);
        setPendingEnterpriseRememberMe(null);
      } finally {
        enterpriseLogoutInProgressRef.current = false;
      }
    },
    [clearEnterpriseStorage, getRememberMe, clearRememberedCredentials]
  );

  const handleDriverLoginSuccess = useCallback(
    async (response: AuthResponse) => {
      pushSessionEvent("LOGIN_SUCCESS");
      // ✅ Les tokens sont déjà stockés dans loginDriver() (SecureStore + AsyncStorage)
      setDriverToken(response.token);
      pushSessionEvent("TOKEN_STORED");
      await setActiveAuthNamespace({
        role: "driver",
        userId: response.user?.public_id || "unknown",
        tenantId: null,
        sessionId: null,
      });
      await storeMode("driver");
      // ✅ P1 strict: rendre l'auth "ready" avant tout appel protégé (ex: /driver/me/profile)
      // Sinon l'intercepteur rejette avec AUTH_NOT_READY et on se retrouve avec un faux "login failed".
      notifyAuthReady();
      // ✅ Éviter race web : attendre que le token soit lisible (SecureStore/AsyncStorage peut être asynchrone)
      // Sinon l'intercepteur peut lire null et rejeter la requête → GET /driver/me/profile jamais envoyé.
      for (let w = 0; w < 15; w++) {
        const tok = await secureStorage.getAccessToken();
        if (tok) break;
        await new Promise((r) => setTimeout(r, 50));
      }
      try {
        // Petit retry pour les devices lents / race condition de propagation (SecureStore/React state)
        let profile: Driver | null = null;
        for (let attempt = 0; attempt < 3; attempt++) {
          try {
            // eslint-disable-next-line no-await-in-loop
            profile = await fetchDriverProfile();
            break;
          } catch (e) {
            if (!isAuthNotReadyError(e)) {
              throw e;
            }
            // eslint-disable-next-line no-await-in-loop
            await new Promise((r) => setTimeout(r, 150 * (attempt + 1)));
          }
        }
        if (!profile) {
          throw new Error("Profil chauffeur indisponible (AUTH_NOT_READY)");
        }
        setDriver(profile);
        await setActiveAuthNamespace({
          role: "driver",
          userId: profile.id || "unknown",
          tenantId: null,
          sessionId: null,
        });
        // ✅ Stocker driver_id pour navigation rapide
        await asyncStorage.setDriverId(profile.id);
      } catch (error) {
        log.warn("driver profile fetch failed", { error });
        // ✅ Important: AUTH_NOT_READY est transitoire → ne pas effacer les tokens / ne pas logout
        if (isAuthNotReadyError(error)) {
          return;
        }
        // P1.B: Ne logout que sur 401/403 (auth invalide). Réseau/5xx => conserver tokens, erreur UI.
        if (isHttpAuthError(error)) {
          await invokeForceLogoutDriver({
            reason: "login_profile_failed",
            severity: "AUTH_HARD_FAILURE",
            source: "driver",
            trigger_source: "manual_action",
          });
        }
        throw error;
      }
    },
    [forceLogoutDriverInternal, storeMode]
  );

  const handleEnterpriseSuccess = useCallback(
    async (
      payload: EnterpriseTokenPayload,
      options?: { skipModeUpdate?: boolean }
    ) => {
      const session = parseEnterpriseSuccess(payload);
      await setActiveAuthNamespace({
        role: "enterprise",
        userId: session.user?.public_id || "unknown",
        tenantId: session.company?.id ?? null,
        sessionId: session.sessionId ?? null,
      });
      setEnterpriseSession(session);
      setPendingEnterpriseMfa(null);

      // Vérifier que le token est bien présent et valide
      if (!session.token) {
        log.error("enterprise token missing in login response");
        throw new Error("Token manquant dans la réponse de login");
      }

      // ✅ CORRECTION : Utiliser SecureStore pour les tokens (sécurisé + cache)
      await secureStorage.setEnterpriseToken(session.token);
      if (session.refreshToken) {
        await secureStorage.setEnterpriseRefreshToken(session.refreshToken);
      } else {
        await secureStorage.removeEnterpriseRefreshToken();
      }

      // Garder AsyncStorage uniquement pour la session complète (données non sensibles)
      await AsyncStorage.multiSet([
        [ENTERPRISE_SESSION_KEY, JSON.stringify(session)],
        // Marquer que la session vient d'être créée pour éviter la vérification immédiate
        ["enterprise_session_just_created", "true"],
      ]);
      // ✅ Ne pas écraser le mode lors d'une restauration bootstrap (évite redirection driver → enterprise)
      if (!options?.skipModeUpdate) {
        await storeMode("enterprise");
      }
      // ✅ P1 strict: rendre l'auth "ready" avant que des requêtes enterprise partent
      notifyAuthReady();

      // Vérifier que le token a bien été stocké
      const storedToken = await secureStorage.getEnterpriseToken();
      if (storedToken !== session.token) {
        log.error("stored token mismatch");
      }

      // Attendre un peu pour s'assurer que AsyncStorage a bien écrit les données
      // avant que d'autres requêtes ne soient faites
      await new Promise(resolve => setTimeout(resolve, 100));

      log.success("enterprise session stored", {
        hasToken: !!session.token,
        hasRefreshToken: !!session.refreshToken,
        userId: session.user?.id,
        companyId: session.company?.id,
        tokenLength: session.token.length,
        tokenStored: storedToken === session.token,
      });
    },
    [storeMode]
  );

  useEffect(() => {
    let isMounted = true;
    let bootstrapStoredMode: string | null = null;
    let enterpriseRestored = false;
    /** Session chauffeur réellement restaurée (profil chargé). Évite notifyAuthReady() en finally quand le driver n'a pas de token. */
    let driverSessionRestored = false;
    const runBootstrap = async () => {
      pushSessionEvent("APP_START");
      try {
        // ⚡ OPTIMISATION Phase 2 : Lecture parallèle des tokens et données de stockage
        // ✅ CORRECTION : Lire aussi les tokens Enterprise en parallèle pour éviter les race conditions
        // Réduit le temps de démarrage et évite les "missing token" au cold start
        const [
          storedMode,
          storedDevice,
          driverRefreshToken,
          driverAccessToken,
          enterpriseToken,
          enterpriseRefreshToken,
          enterpriseSessionRaw,
        ] = await Promise.all([
          AsyncStorage.getItem(MODE_KEY),
          AsyncStorage.getItem(ENTERPRISE_DEVICE_KEY),
          secureStorage.getRefreshToken(), // Driver refresh token
          secureStorage.getAccessToken(), // Driver access token
          secureStorage.getEnterpriseToken(), // Enterprise access token
          secureStorage.getEnterpriseRefreshToken(), // Enterprise refresh token
          AsyncStorage.getItem(ENTERPRISE_SESSION_KEY), // Enterprise session (données non sensibles)
        ]);
        bootstrapStoredMode = storedMode;
        if (isDebugAuthEnabled()) {
          debugAuthLog("boot_storage", {
            has_ent_refresh: enterpriseRefreshToken ? 1 : 0,
            len: enterpriseRefreshToken?.length ?? 0,
          });
        }

        // Traiter le mode stocké
        if (storedMode === "driver" || storedMode === "enterprise") {
          setModeState(storedMode);
        } else {
          await AsyncStorage.setItem(MODE_KEY, "enterprise");
        }

        // Traiter le device ID
        if (storedDevice) setDeviceId(storedDevice);

        // ✅ Auto-login : essayer d'abord l'access token, puis le refresh token
        // Ne se déclenche que si on est en mode "driver" ou si le mode n'est pas encore défini
        if (storedMode === "driver" || !storedMode || storedMode === null) {
          let profileLoaded = false;

          // Marquer comme en cours de chargement pour éviter la navigation prématurée
          if (isMounted) {
            setDriverLoading(true);
          }

          try {
            // 1. Essayer d'abord avec l'access token (priorité après un switch)
            // ⚡ OPTIMISATION : driverAccessToken déjà lu en parallèle ci-dessus
            if (driverAccessToken) {
              try {
                setDriverToken(driverAccessToken);
                // ✅ Important: rendre l'auth prête avant d'appeler un endpoint protégé
                notifyAuthReady();
                const profile = await fetchDriverProfile();
                if (isMounted) {
                  setDriver(profile);
                  await setActiveAuthNamespace({
                    role: "driver",
                    userId: profile.id || "unknown",
                    tenantId: null,
                    sessionId: null,
                  });
                  await asyncStorage.setDriverId(profile.id);
                  profileLoaded = true;
                  driverSessionRestored = true;
                  // S'assurer qu'on est en mode driver
                  await storeMode("driver");
                }
              } catch (error) {
                log.warn("driver access token invalid, trying refresh", { error });
                // Continuer avec le refresh token si l'access token échoue
              }
            }

            // 2. Si l'access token n'a pas fonctionné, essayer le refresh token
            if (!profileLoaded && driverRefreshToken) {
              try {
                // ✅ Utiliser le singleflight driver (cohérence + évite races)
                const newAccessToken = await refreshDriverTokenOrchestrated("boot_restore");
                setDriverToken(newAccessToken);
                notifyAuthReady();

                // S'assurer qu'on est en mode driver
                await storeMode("driver");

                // Charger le profil driver
                const profile = await fetchDriverProfile();
                if (isMounted) {
                  setDriver(profile);
                  await setActiveAuthNamespace({
                    role: "driver",
                    userId: profile.id || "unknown",
                    tenantId: null,
                    sessionId: null,
                  });
                  await asyncStorage.setDriverId(profile.id);
                  profileLoaded = true;
                  driverSessionRestored = true;
                }
              } catch (refreshError) {
                log.warn("auto-login failed (refresh token)", { error: refreshError });

                // ✅ P0.2.B : Offline/timeout/5xx ≠ logout — uniquement 401/403 invalide la session
                if (isNetworkError(refreshError)) {
                  log.warn("boot network error, tokens kept");
                  // Pas de forceLogout, pas de wipe storage
                } else if (isHttpAuthError(refreshError)) {
                  const status = getHttpStatus(refreshError);
                  await invokeForceLogoutDriver({
                    reason: status === 401 ? "refresh_invalid" : "account_disabled",
                    severity: "AUTH_HARD_FAILURE",
                    source: "driver",
                    trigger_source: "bootstrap",
                  });
                } else {
                  // 5xx, autre → pas de logout
                  log.warn("boot server error, tokens kept");
                }

                // ✅ PHASE 2 : Si refresh a échoué (hors auth invalid, on a déjà logout), essayer biométrique
                if (!profileLoaded && !isHttpAuthError(refreshError)) {
                  try {
                    const biometricSuccess = await autoLoginWithBiometric({
                      promptMessage: "Authentifiez-vous pour vous reconnecter",
                      cancelLabel: "Annuler",
                      disableDeviceFallback: false,
                      fallbackLabel: "Utiliser le code PIN",
                    });

                    if (biometricSuccess) {
                      notifyAuthReady();
                      const profile = await fetchDriverProfile();
                      if (isMounted) {
                        setDriver(profile);
                        await setActiveAuthNamespace({
                          role: "driver",
                          userId: profile.id || "unknown",
                          tenantId: null,
                          sessionId: null,
                        });
                        await asyncStorage.setDriverId(profile.id);
                        await storeMode("driver");
                        profileLoaded = true;
                        driverSessionRestored = true;
                        log.success("auto-login succeeded with biometric");
                      }
                    } else {
                      log.debug("biometric auto-login cancelled or failed");
                    }
                  } catch (autoLoginError) {
                    if (autoLoginError instanceof BiometricNoCredentialsError) {
                      if (isMounted) {
                        Alert.alert(
                          "Reconnexion requise",
                          "Identifiants expirés ou indisponibles. Veuillez vous reconnecter.",
                          [{ text: "Se connecter", style: "default" }]
                        );
                      }
                    } else {
                      log.warn("biometric auto-login failed", { error: autoLoginError });
                    }
                    // ✅ P0.2.B : Pas de forceLogout sur erreur réseau ou autre
                    if (isMounted && isHttpAuthError(autoLoginError)) {
                      const status = getHttpStatus(autoLoginError);
                      await invokeForceLogoutDriver({
                        reason:
                          status === 401 ? "refresh_invalid" : "account_disabled",
                        severity: "AUTH_HARD_FAILURE",
                        source: "driver",
                        trigger_source: "bootstrap",
                      });
                    }
                  }
                  // Si toujours pas de profil chargé : pas de forceLogout (tokens conservés pour retry)
                }
              }
            }

            // 3. Si aucun token n'a fonctionné, essayer auto-login avec authentification biométrique
            if (!profileLoaded && !driverAccessToken && !driverRefreshToken) {
              try {
                const biometricSuccess = await autoLoginWithBiometric({
                  promptMessage: "Authentifiez-vous pour vous reconnecter",
                  cancelLabel: "Annuler",
                  disableDeviceFallback: false,
                  fallbackLabel: "Utiliser le code PIN",
                });

                if (biometricSuccess) {
                  notifyAuthReady();
                  const profile = await fetchDriverProfile();
                  if (isMounted) {
                    setDriver(profile);
                    await setActiveAuthNamespace({
                      role: "driver",
                      userId: profile.id || "unknown",
                      tenantId: null,
                      sessionId: null,
                    });
                    await asyncStorage.setDriverId(profile.id);
                    await storeMode("driver");
                    profileLoaded = true;
                    driverSessionRestored = true;
                    log.success("auto-login succeeded with biometric");
                  }
                } else {
                  // ✅ P0.2.B : User annulé → pas de forceLogout (rien à effacer)
                }
              } catch (autoLoginError) {
                if (autoLoginError instanceof BiometricNoCredentialsError) {
                  if (isMounted) {
                    Alert.alert(
                      "Reconnexion requise",
                      "Identifiants expirés ou indisponibles. Veuillez vous reconnecter.",
                      [{ text: "Se connecter", style: "default" }]
                    );
                  }
                } else {
                  log.warn("biometric auto-login failed", { error: autoLoginError });
                }
                // ✅ P0.2.B : Pas de forceLogout sur erreur réseau ; uniquement 401/403
                if (isMounted && isHttpAuthError(autoLoginError)) {
                  const status = getHttpStatus(autoLoginError);
                  await invokeForceLogoutDriver({
                    reason: status === 401 ? "refresh_invalid" : "account_disabled",
                    severity: "AUTH_HARD_FAILURE",
                    source: "driver",
                    trigger_source: "bootstrap",
                  });
                }
              }
            }
          } finally {
            // Toujours arrêter le chargement à la fin
            if (isMounted) {
              setDriverLoading(false);
            }
          }
        }

        // ✅ CORRECTION : Utiliser les tokens Enterprise déjà lus en parallèle au début
        // Évite une deuxième lecture et réduit les race conditions
        const justCreated = await AsyncStorage.getItem("enterprise_session_just_created");

        if (enterpriseToken && enterpriseSessionRaw) {
          try {
            enterpriseRestored = true;
            const parsed: EnterpriseSessionState =
              JSON.parse(enterpriseSessionRaw);
            // Restaurer la session depuis le stockage
            // ✅ CORRECTION : Utiliser enterpriseToken déjà lu (depuis SecureStore)
            setEnterpriseSession({
              ...parsed,
              token: enterpriseToken,
              refreshToken: enterpriseRefreshToken ?? parsed.refreshToken ?? null,
            });
            // ✅ P1 : Notifier auth ready dès que la session entreprise est restaurée
            // Évite "missing_refresh_token" : les requêtes ne partent qu'une fois les tokens chargés
            notifyAuthReady();

            // Si la session vient d'être créée (juste après un login), ne pas la vérifier immédiatement
            // Supprimer le flag pour les prochaines fois
            if (justCreated === "true") {
              await AsyncStorage.removeItem("enterprise_session_just_created");
              log.info("enterprise session just created, verification deferred");
              return; // Sortir sans vérifier la session
            }

            // Vérifier la session en arrière-plan (non bloquant)
            // Si la vérification échoue, on essaiera de rafraîchir le token
            (async () => {
              try {
                const latest = await fetchEnterpriseSession(enterpriseToken);
                // ✅ CORRECTION : Utiliser SecureStore pour le refresh token
                const refreshToken = await secureStorage.getEnterpriseRefreshToken() ?? parsed.refreshToken ?? null;
                const updated: EnterpriseSessionState = {
                  token: enterpriseToken,
                  refreshToken,
                  user: latest.user,
                  company: {
                    id: latest.company.id,
                    name: latest.company.name,
                    dispatchMode:
                      (latest.company as any).dispatchMode ??
                      latest.company.dispatch_mode,
                  },
                  scopes: latest.scopes ?? [],
                  sessionId: latest.session_id,
                };
                setEnterpriseSession(updated);
                await AsyncStorage.setItem(
                  ENTERPRISE_SESSION_KEY,
                  JSON.stringify(updated)
                );
              } catch (sessionError: any) {
                // Si la vérification échoue avec un 401, essayer de rafraîchir le token
                if (sessionError?.response?.status === 401) {
                  // ✅ CORRECTION : Utiliser SecureStore pour le refresh token
                  const refreshToken = await secureStorage.getEnterpriseRefreshToken() ?? parsed.refreshToken;
                  if (refreshToken) {
                    try {
                      const refreshResponse = await runEnterpriseRefreshSingleflight(
                        buildEnterpriseSessionKeyFromState({
                          ...parsed,
                          token: enterpriseToken,
                        } as EnterpriseSessionState),
                        "bootstrap",
                        refreshToken
                      );
                      await handleEnterpriseSuccess(refreshResponse, {
                        skipModeUpdate: true,
                      });
                    } catch (refreshError) {
                      log.warn("enterprise token refresh failed", { error: refreshError });
                      // Ne pas nettoyer la session ici - laisser l'utilisateur utiliser l'app
                      // La session sera invalidée lors de la prochaine requête API réelle
                    }
                  } else {
                    log.warn("enterprise session invalid, no refresh token", { error: sessionError });
                    // Ne pas nettoyer immédiatement - laisser l'utilisateur utiliser l'app
                  }
                } else {
                  log.warn("enterprise session verification failed", { error: sessionError });
                  // Ne pas nettoyer la session pour les erreurs réseau temporaires
                }
              }
            })();
          } catch (error) {
            log.warn("enterprise session restore failed", { error });
            enterpriseRestored = false;
            await invokeForceLogoutEnterprise({
              reason: "refresh_invalid",
              severity: "AUTH_HARD_FAILURE",
              source: "enterprise",
              trigger_source: "bootstrap",
            });
          }
        }
      } finally {
        if (isMounted) {
          setInitialLoading(false);
          // ✅ P1 : Ne notifier auth ready que si une session valide a été restaurée pour le mode actuel.
          // Évite AUTH_NOT_READY missing_access_token (driver) et missing_refresh_token (enterprise).
          const isEnterpriseWithoutSession =
            bootstrapStoredMode === "enterprise" && !enterpriseRestored;
          const isDriverWithoutSession =
            (bootstrapStoredMode === "driver" || !bootstrapStoredMode) && !driverSessionRestored;
          const shouldNotifyAuthReady =
            !isEnterpriseWithoutSession && !isDriverWithoutSession;
          if (isDebugAuthEnabled()) {
            debugAuthLog("notify_auth_ready", {
              mode: bootstrapStoredMode ?? undefined,
              enterprise_restored: enterpriseRestored ? 1 : 0,
              driver_session_restored: driverSessionRestored ? 1 : 0,
              should_notify: shouldNotifyAuthReady ? 1 : 0,
            });
          }
          if (shouldNotifyAuthReady) {
            notifyAuthReady();
          }
        }
      }
    };

    const BOOTSTRAP_TIMEOUT_MS = 25000;

    const executeBootstrapOnce = async () => {
      // Ne jamais skip : hot reload / StrictMode remount réinitialise l'état React
      // mais les variables module persistent → on doit toujours re-exécuter pour restaurer depuis storage
      if (!authBootstrapOncePromise) {
        authBootstrapOncePromise = runBootstrap();
      } else {
        log.info("auth bootstrap already running, reusing global promise");
      }

      const timeoutPromise = new Promise<void>((resolve) => {
        setTimeout(() => {
          log.warn("bootstrap timeout, releasing loading to avoid infinite spinner");
          resolve();
        }, BOOTSTRAP_TIMEOUT_MS);
      });

      try {
        await Promise.race([authBootstrapOncePromise, timeoutPromise]);
      } catch (e) {
        log.warn("bootstrap error", { error: e });
      } finally {
        // Timeout ou succès : toujours sortir du loading pour éviter boucle infinie
        if (isMounted) {
          setInitialLoading(false);
        }
      }
    };

    void executeBootstrapOnce();
    return () => {
      isMounted = false;
      // StrictMode / hot reload remount : reset pour re-exécuter le bootstrap et restaurer mode/driver depuis storage
      authBootstrapOncePromise = null;
    };
  }, [clearDriverStorage, clearEnterpriseStorage, storeMode, handleEnterpriseSuccess]);

  // ✅ PHASE 3 : REFRESH PROACTIF amélioré : Rafraîchir le token driver 10 minutes avant expiration
  // P0.3.A : Cooldown + backoff sur échec (évite boucles, battery drain)
  useEffect(() => {
    if (!driverToken || mode !== "driver") return;

    const expiresAt = getTokenExpiration(driverToken);
    if (!expiresAt) {
      log.warn("driver token expiration decode failed");
      return;
    }

    const now = Date.now();
    const timeUntilExpiry = expiresAt - now;
    const refreshBeforeExpiry = 10 * 60 * 1000; // 10 minutes

    const runProactiveRefresh = () => {
      // ✅ P0.3.A : Vérifier cooldown avant tentative
      if (isDriverProactiveRefreshInCooldown()) {
        const remaining = getDriverProactiveRefreshCooldownRemaining();
        log.warn("driver proactive refresh in cooldown", { remainingSeconds: Math.round(remaining / 1000) });
        driverProactiveRefreshTimeoutRef.current = setTimeout(runProactiveRefresh, remaining);
        return;
      }

      (async () => {
        log.info("driver proactive refresh started");

        const MAX_RETRIES = 3;
        let retryCount = 0;
        let lastError: any = null;

        while (retryCount < MAX_RETRIES) {
          try {
            const refreshToken = await secureStorage.getRefreshToken();
            if (!refreshToken) {
              throw new Error("Pas de refresh token disponible");
            }

            const newAccessToken = await refreshDriverTokenOrchestrated("proactive_refresh");
            setDriverToken(newAccessToken);
            resetDriverProactiveRefreshCooldown();

            invalidateInterceptorCache();

            log.success("driver proactive refresh succeeded", retryCount > 0 ? { retryCount } : undefined);
            return;
          } catch (error: any) {
            lastError = error;
            const status = error?.response?.status;
            const isNetworkError = !error?.response; // Pas de réponse = erreur réseau

            // ✅ PHASE 3 : Ne pas retry pour les erreurs critiques (401, 403)
            if (status === 401 || status === 403) {
              log.error("driver proactive refresh failed", { status, data: error?.response?.data, message: error?.message });
              break; // Sortir de la boucle, ne pas retry
            }

            // ✅ PHASE 3 : Retry uniquement pour erreurs réseau ou serveur (500, etc.)
            if (isNetworkError || (status && status >= 500)) {
              retryCount++;
              if (retryCount < MAX_RETRIES) {
                const delay = Math.min(1000 * Math.pow(2, retryCount - 1), 10000); // Backoff exponentiel (max 10s)
                log.warn("driver proactive refresh retry", { attempt: retryCount, maxRetries: MAX_RETRIES, delayMs: delay });
                await new Promise(resolve => setTimeout(resolve, delay));
                continue; // Retry
              } else {
                log.error("driver proactive refresh failed after retries", { maxRetries: MAX_RETRIES });
                break; // Toutes les tentatives ont échoué
              }
            } else {
              // Autres erreurs (400, etc.) → ne pas retry
              log.warn("driver proactive refresh failed, no retry", { status });
              break;
            }
          }
        }

        // ✅ P0.2.A : Si toutes les tentatives ont échoué, vérifier si logout nécessaire
        if (lastError) {
          const status = lastError?.response?.status;
          const errorData = lastError?.response?.data;

          // ✅ P0.2.A : 401/403 mais access encore valide → best effort, pas de logout
          if ((status === 401 || status === 403) && accessStillValid(driverToken)) {
            log.warn("driver proactive refresh failed but access still valid", { status });
            const backoff = recordDriverProactiveRefreshFailure();
            driverProactiveRefreshTimeoutRef.current = setTimeout(runProactiveRefresh, backoff);
            return;
          }

          // ✅ Si 403 (compte désactivé) et access expiré, forcer déconnexion
          if (status === 403) {
            log.error("driver account disabled on proactive refresh", { errorData });
            await invokeForceLogoutDriver({
              reason: "account_disabled",
              severity: "AUTH_HARD_FAILURE",
              source: "driver",
              trigger_source: "foreground_resume",
            });
            return;
          }

          // 401 et access expiré, ou 5xx/network → cooldown + retry
          const backoff = recordDriverProactiveRefreshFailure();
          driverProactiveRefreshTimeoutRef.current = setTimeout(runProactiveRefresh, backoff);
          log.warn("driver proactive refresh failed, fallback to interceptor", { error: lastError });
        }
      })();
    };

    // ✅ PHASE 3 : Si le token expire dans plus de 10 minutes, planifier le refresh
    if (timeUntilExpiry > refreshBeforeExpiry) {
      const initialDelay = isDriverProactiveRefreshInCooldown()
        ? getDriverProactiveRefreshCooldownRemaining()
        : timeUntilExpiry - refreshBeforeExpiry;
      const timeoutId = setTimeout(runProactiveRefresh, initialDelay);
      driverProactiveRefreshTimeoutRef.current = timeoutId;

      log.info("driver proactive refresh scheduled", { minutes: Math.round(initialDelay / 1000 / 60) });

      return () => {
        if (driverProactiveRefreshTimeoutRef.current) {
          clearTimeout(driverProactiveRefreshTimeoutRef.current);
          driverProactiveRefreshTimeoutRef.current = null;
        }
      };
    } else if (timeUntilExpiry > 0 && timeUntilExpiry <= refreshBeforeExpiry) {
      // ✅ PHASE 3 : Token expire bientôt (< 10min), rafraîchir immédiatement avec retry
      log.info("driver token expiring soon, immediate refresh");
      (async () => {
        const MAX_RETRIES = 3;
        let retryCount = 0;
        let lastError: any = null;

        while (retryCount < MAX_RETRIES) {
          try {
            const newAccessToken = await refreshDriverTokenOrchestrated("proactive_refresh");
            setDriverToken(newAccessToken);
            invalidateInterceptorCache();
            resetDriverProactiveRefreshCooldown();
            log.success("driver immediate refresh succeeded", retryCount > 0 ? { retryCount } : undefined);
            return;
          } catch (error: any) {
            lastError = error;
            const status = error?.response?.status;
            const isNetworkError = !error?.response;

            if (status === 401 || status === 403) {
              log.error("driver immediate refresh failed", { status });
              break;
            }

            if (isNetworkError || (status && status >= 500)) {
              retryCount++;
              if (retryCount < MAX_RETRIES) {
                const delay = Math.min(1000 * Math.pow(2, retryCount - 1), 10000);
                log.warn("driver immediate refresh retry", { attempt: retryCount, maxRetries: MAX_RETRIES, delayMs: delay });
                await new Promise(resolve => setTimeout(resolve, delay));
                continue;
              }
            } else {
              break;
            }
          }
        }

        // ✅ P0.2.A : Gérer les erreurs finales (refresh immédiat = token proche expiration)
        if (lastError) {
          const status = lastError?.response?.status;
          const errorData = lastError?.response?.data;

          // ✅ P0.2.A : 401/403 mais access encore valide (skew horloge) → pas de logout
          if ((status === 401 || status === 403) && accessStillValid(driverToken)) {
            log.warn("driver immediate refresh failed but access still valid", { status });
            recordDriverProactiveRefreshFailure();
            return;
          }

          if (status === 403) {
            log.error("driver account disabled on immediate refresh", { errorData });
            await invokeForceLogoutDriver({
              reason: "account_disabled",
              severity: "AUTH_HARD_FAILURE",
              source: "driver",
              trigger_source: "proactive_refresh",
            });
            return;
          }

          recordDriverProactiveRefreshFailure();
          log.warn("driver immediate refresh failed", { error: lastError });
        }
      })();
    }
  }, [driverToken, mode, forceLogoutDriverInternal]);

  // ✅ PHASE 3 + P0.4 : Refresh et FOREGROUND_RESYNC au retour au premier plan
  // 1) recharger tokens si besoin 2) ping /driver/me avec retry court 3) reconnect socket
  // Jamais de déconnexion après 30–60 min en arrière-plan
  useEffect(() => {
    if (!driverToken || mode !== "driver") return;

    const handleAppStateChange = async (nextAppState: AppStateStatus) => {
      if (nextAppState === "active") {
        pushSessionEvent("APP_FOREGROUND");
        pushSessionEvent("FOREGROUND_RESYNC_START");
        // Différer le resync lourd pour laisser la transition foreground se terminer (évite ANR)
        InteractionManager.runAfterInteractions(() => {
        setTimeout(async () => {
        let resyncOk = true;
        try {
          // 1) Recharger tokens depuis storage et refresh si expiré / proche expiration
          const expiresAt = getTokenExpiration(driverToken);
          if (!expiresAt) {
            const refreshToken = await secureStorage.getRefreshToken();
            if (refreshToken) {
              try {
                const newAccessToken = await refreshDriverTokenOrchestrated("foreground_resync");
                setDriverToken(newAccessToken);
                invalidateInterceptorCache();
              } catch (_e) {
                // continuer, l’intercepteur gérera
              }
            }
          } else {
            const timeUntilExpiry = expiresAt - Date.now();
            const refreshThreshold = 15 * 60 * 1000;
            if (timeUntilExpiry <= 0 || timeUntilExpiry < refreshThreshold) {
              const refreshToken = await secureStorage.getRefreshToken();
              if (refreshToken) {
                try {
                  const newAccessToken = await refreshDriverTokenOrchestrated("foreground_resync");
                  setDriverToken(newAccessToken);
                  invalidateInterceptorCache();
                } catch (_e) {
                  // ne pas déconnecter
                }
              }
            }
          }

          // 2) Ping /driver/me avec retry court (2 tentatives)
          const tokenForPing = await secureStorage.getAccessToken();
          if (tokenForPing) {
            for (let attempt = 1; attempt <= 2; attempt++) {
              try {
                await fetchDriverProfile();
                break;
              } catch (e) {
                if (attempt === 2) {
                  resyncOk = false;
                  log.warn("foreground resync ping failed after 2 attempts");
                }
              }
            }
          }

          // 3) Reconnect socket (resubscribe fait dans connectSocket)
          const tokenForSocket = await secureStorage.getAccessToken();
          if (tokenForSocket) {
            connectSocket(tokenForSocket, "driver").catch(() => { });
          }
        } finally {
          pushSessionEvent(resyncOk ? "FOREGROUND_RESYNC_SUCCESS" : "FOREGROUND_RESYNC_FAIL");
        }
        }, 200);
        });
      } else {
        // Différer pour éviter ANR lors de la transition background (low memory)
        InteractionManager.runAfterInteractions(() => {
          setTimeout(() => pushSessionEvent("APP_BACKGROUND"), 0);
        });
      }
    };

    const subscription = AppState.addEventListener("change", handleAppStateChange);
    return () => subscription.remove();
  }, [driverToken, mode]);

  // ✅ FIX iOS : Foreground resync pour le mode enterprise
  // iOS suspend les timers JS en arrière-plan → le refresh proactif ne se déclenche pas.
  // Sans ce handler, le token enterprise (45min) expire et l'utilisateur est déconnecté au retour.
  useEffect(() => {
    if (!enterpriseSession?.token || mode !== "enterprise") return;

    const handleEnterpriseAppStateChange = async (nextAppState: AppStateStatus) => {
      if (nextAppState === "active") {
        pushSessionEvent("ENTERPRISE_APP_FOREGROUND");
        InteractionManager.runAfterInteractions(async () => {
        try {
          const expiresAt = getTokenExpiration(enterpriseSession.token);
          const refreshThreshold = 10 * 60 * 1000; // 10 minutes
          const needsRefresh =
            !expiresAt ||
            (expiresAt - Date.now()) <= 0 ||
            (expiresAt - Date.now()) < refreshThreshold;

          if (needsRefresh) {
            const refreshToken =
              enterpriseSession.refreshToken ||
              (await secureStorage.getEnterpriseRefreshToken());
            if (refreshToken) {
              try {
                const refreshResponse = await runEnterpriseRefreshSingleflight(
                  buildEnterpriseSessionKeyFromState(enterpriseSession),
                  "foreground_resume",
                  refreshToken
                );
                await handleEnterpriseSuccess(refreshResponse);
                invalidateEnterpriseInterceptorCache();
                log.success("enterprise foreground resync refreshed token");
              } catch (_e) {
                log.warn("enterprise foreground resync refresh failed, interceptor will handle");
              }
            }
          }
        } catch (e) {
          log.warn("enterprise foreground resync error", { error: e });
        }
        });
      }
    };

    const subscription = AppState.addEventListener("change", handleEnterpriseAppStateChange);
    return () => subscription.remove();
  }, [enterpriseSession?.token, mode, handleEnterpriseSuccess]);

  // ✅ REFRESH PROACTIF : Rafraîchir le token entreprise 5 minutes avant expiration
  useEffect(() => {
    if (!enterpriseSession?.token || mode !== "enterprise") return;

    const expiresAt = getTokenExpiration(enterpriseSession.token);
    if (!expiresAt) {
      log.warn("enterprise token expiration decode failed");
      return;
    }

    const now = Date.now();
    const timeUntilExpiry = expiresAt - now;
    const refreshBeforeExpiry = 5 * 60 * 1000; // 5 minutes

    if (timeUntilExpiry > refreshBeforeExpiry) {
      const runEnterpriseProactiveRefresh = () => {
        if (isEnterpriseProactiveRefreshInCooldown()) {
          const remaining = getEnterpriseProactiveRefreshCooldownRemaining();
          log.warn("enterprise proactive refresh in cooldown", { remainingSeconds: Math.round(remaining / 1000) });
          enterpriseProactiveRefreshTimeoutRef.current = setTimeout(runEnterpriseProactiveRefresh, remaining);
          return;
        }

        (async () => {
          log.info("enterprise proactive refresh started");
          try {
            // ✅ CORRECTION : Utiliser SecureStore pour le refresh token
            const refreshToken = enterpriseSession.refreshToken || await secureStorage.getEnterpriseRefreshToken();
            if (!refreshToken) {
              log.warn("enterprise proactive refresh no refresh token");
              return;
            }

            log.debug("enterprise proactive refresh token available", { length: refreshToken.length });

            const refreshResponse = await runEnterpriseRefreshSingleflight(
              buildEnterpriseSessionKeyFromState(enterpriseSession),
              "proactive_refresh",
              refreshToken
            );
            await handleEnterpriseSuccess(refreshResponse);

            // ⚡ CORRECTION : Invalider le cache interceptor pour forcer l'utilisation du nouveau token
            invalidateEnterpriseInterceptorCache();

            log.success("enterprise proactive refresh succeeded");
          } catch (error: any) {
            const status = error?.response?.status;
            const errorData = error?.response?.data;
            const errorMessage = errorData?.error || error?.message;

            // ✅ Si 401 (refresh token invalide/expiré), ne pas déconnecter immédiatement
            // L'intercepteur gérera la déconnexion lors de la prochaine requête
            if (status === 401) {
              log.error("enterprise proactive refresh 401", { errorMessage });
              // Ne pas déconnecter ici, l'intercepteur gérera lors de la prochaine requête réelle
              return;
            }

            // ✅ P0.2.A : 403 mais access encore valide → best effort, pas de logout
            if (status === 403 && enterpriseSession && accessStillValid(enterpriseSession.token)) {
              log.warn("enterprise proactive refresh failed but access still valid");
              const backoff = recordEnterpriseProactiveRefreshFailure();
              enterpriseProactiveRefreshTimeoutRef.current = setTimeout(runEnterpriseProactiveRefresh, backoff);
              return;
            }

            if (status === 403) {
              log.error("enterprise account disabled on proactive refresh", { errorData });
              await invokeForceLogoutEnterprise({
                reason: "account_disabled",
                severity: "AUTH_HARD_FAILURE",
                source: "enterprise",
                trigger_source: "foreground_resume",
              });
              return;
            }

            // Autres erreurs (réseau, serveur, etc.) → cooldown + retry
            const backoff = recordEnterpriseProactiveRefreshFailure();
            enterpriseProactiveRefreshTimeoutRef.current = setTimeout(runEnterpriseProactiveRefresh, backoff);
            log.warn("enterprise proactive refresh failed", { status: status || "network", errorMessage });
          }
        })();
      };

      const initialDelay = isEnterpriseProactiveRefreshInCooldown()
        ? getEnterpriseProactiveRefreshCooldownRemaining()
        : timeUntilExpiry - refreshBeforeExpiry;
      const timeoutId = setTimeout(runEnterpriseProactiveRefresh, initialDelay);
      enterpriseProactiveRefreshTimeoutRef.current = timeoutId;

      log.info("enterprise proactive refresh scheduled", { minutes: Math.round(initialDelay / 1000 / 60) });

      return () => {
        if (enterpriseProactiveRefreshTimeoutRef.current) {
          clearTimeout(enterpriseProactiveRefreshTimeoutRef.current);
          enterpriseProactiveRefreshTimeoutRef.current = null;
        }
      };
    } else if (timeUntilExpiry > 0 && timeUntilExpiry <= refreshBeforeExpiry) {
      // Token expire bientôt (< 5min), rafraîchir immédiatement
      log.info("enterprise token expiring soon, immediate refresh");
      (async () => {
        try {
          // ✅ CORRECTION : Utiliser SecureStore pour le refresh token
          const refreshToken = enterpriseSession.refreshToken || await secureStorage.getEnterpriseRefreshToken();
          if (refreshToken) {
            const refreshResponse = await runEnterpriseRefreshSingleflight(
              buildEnterpriseSessionKeyFromState(enterpriseSession),
              "proactive_refresh",
              refreshToken
            );
            await handleEnterpriseSuccess(refreshResponse);

            // ⚡ CORRECTION : Invalider le cache interceptor pour forcer l'utilisation du nouveau token
            invalidateEnterpriseInterceptorCache();
            resetEnterpriseProactiveRefreshCooldown();

            log.success("enterprise immediate refresh succeeded");
          }
        } catch (error: any) {
          const status = error?.response?.status;
          const errorData = error?.response?.data;

          // ✅ P0.2.A : 403 mais access encore valide → pas de logout
          if (status === 403 && enterpriseSession && accessStillValid(enterpriseSession.token)) {
            log.warn("enterprise immediate refresh failed but access still valid");
            recordEnterpriseProactiveRefreshFailure();
            return;
          }

          if (status === 403) {
            log.error("enterprise account disabled on immediate refresh", { errorData });
            await invokeForceLogoutEnterprise({
              reason: "account_disabled",
              severity: "AUTH_HARD_FAILURE",
              source: "enterprise",
              trigger_source: "proactive_refresh",
            });
            return;
          }

          recordEnterpriseProactiveRefreshFailure();
          log.warn("enterprise immediate refresh failed", { error });
        }
      })();
    }
  }, [enterpriseSession?.token, mode, handleEnterpriseSuccess, forceLogoutEnterpriseInternal]);

  // ✅ Recharger la session entreprise après un switchMode vers "enterprise"
  // Ce useEffect se déclenche quand le mode change vers "enterprise" et qu'il n'y a pas encore de session chargée
  useEffect(() => {

    if (mode !== "enterprise" || enterpriseSession || initialLoading) {
      return;
    }

    let isMounted = true;
    (async () => {
      try {

        // ✅ CORRECTION : Utiliser SecureStore pour le token
        const [enterpriseToken, enterpriseSessionRaw, justCreated] = await Promise.all([
          secureStorage.getEnterpriseToken(),
          AsyncStorage.getItem(ENTERPRISE_SESSION_KEY),
          AsyncStorage.getItem("enterprise_session_just_created"),
        ]);


        if (enterpriseToken && enterpriseSessionRaw && isMounted) {
          try {
            const parsed: EnterpriseSessionState = JSON.parse(enterpriseSessionRaw);
            // Restaurer la session depuis le stockage
            setEnterpriseSession({ ...parsed, token: enterpriseToken });


            // Si la session vient d'être créée (juste après un switch), ne pas la vérifier immédiatement
            if (justCreated === "true") {
              await AsyncStorage.removeItem("enterprise_session_just_created");
              log.info("enterprise session loaded after switch, verification deferred");
            }
          } catch (error) {
            log.warn("enterprise session load error after switch", { error });
          }
        }
      } catch (error) {
        log.warn("enterprise async storage read error after switch", { error });
      }
    })();

    return () => {
      isMounted = false;
    };
  }, [mode, enterpriseSession, initialLoading]);

  const login = useCallback(
    async (email: string, password: string, rememberMe: boolean = false) => {
      setDriverLoading(true);
      try {
        const response = await loginDriver(email, password);
        await handleDriverLoginSuccess(response);

        // ✅ PHASE 1 : Se souvenir de moi — SecureStore uniquement, jamais de log du mot de passe
        if (rememberMe) {
          try {
            await persistRememberMe(true);
            await setRememberedCredentials(email.trim(), password);
          } catch {
            await persistRememberMe(false);
            throw new RememberMeStorageError();
          }
        } else {
          await persistRememberMe(false);
          await clearRememberedCredentials();
        }
      } finally {
        setDriverLoading(false);
      }
    },
    [handleDriverLoginSuccess]
  );

  const logout = useCallback(async () => {
    pushSessionEvent("LOGOUT_TRIGGERED");
    await invokeForceLogoutDriver({
      reason: "manual_logout",
      severity: "AUTH_MANUAL",
      source: "driver",
      trigger_source: "manual_action",
    });
  }, []);

  // ✅ Fonction pour charger la session driver depuis SecureStorage sans faire de requête API
  // Utile après un switchMode quand la session vient d'être créée
  const loadDriverSession = useCallback(async () => {

    setDriverLoading(true);
    try {
      const [accessToken, refreshToken, userPublicId] = await Promise.all([
        secureStorage.getAccessToken(),
        secureStorage.getRefreshToken(),
        secureStorage.getUserPublicId(),
      ]);


      if (accessToken) {
        setDriverToken(accessToken);

        // Charger le profil driver pour mettre à jour le contexte
        try {
          const profile = await fetchDriverProfile();
          setDriver(profile);
          await setActiveAuthNamespace({
            role: "driver",
            userId: profile.id || userPublicId || "unknown",
            tenantId: null,
            sessionId: null,
          });
          await asyncStorage.setDriverId(profile.id);


          log.success("driver session loaded from storage", {
            hasToken: !!accessToken,
            hasRefreshToken: !!refreshToken,
            driverId: profile.id,
          });
        } catch (profileError) {
          log.warn("driver profile load error", { error: profileError });
          // Ne pas nettoyer la session ici, le token est valide mais le profil ne peut pas être chargé
        }
      } else {
        setDriver(null);
        setDriverToken(null);
        log.info("no driver session in storage");
      }
    } catch (error) {
      log.error("driver session load error", { error });
      setDriver(null);
      setDriverToken(null);
    } finally {
      setDriverLoading(false);
    }
  }, []);

  const refreshProfile = useCallback(async () => {
    // Récupérer le token depuis le state ou SecureStorage
    let currentToken = driverToken;
    if (!currentToken) {
      // Si driverToken n'est pas dans le state, essayer de le récupérer depuis SecureStorage
      currentToken = await secureStorage.getAccessToken();
      if (currentToken) {
        setDriverToken(currentToken);
        log.info("driver token restored from storage for refreshProfile");
      } else {
        log.warn("no driver token for refresh profile");
        return;
      }
    }

    setDriverLoading(true);
    try {
      const profile = await fetchDriverProfile();
      setDriver(profile);
      await setActiveAuthNamespace({
        role: "driver",
        userId: profile.id || "unknown",
        tenantId: null,
        sessionId: null,
      });
      await asyncStorage.setDriverId(profile.id);
      log.success("driver profile refreshed", { profileId: profile.id });
    } catch (error: any) {
      const status = error?.response?.status;

      // P1.B: Sur 401/403, tenter refresh+retry avant logout. Réseau/5xx => jamais logout.
      if (isHttpAuthError(error)) {
        try {
          const newToken = await refreshDriverTokenOrchestrated("profile_refresh");
          setDriverToken(newToken);
          invalidateInterceptorCache();
          const profile = await fetchDriverProfile();
          setDriver(profile);
          await setActiveAuthNamespace({
            role: "driver",
            userId: profile.id || "unknown",
            tenantId: null,
            sessionId: null,
          });
          await asyncStorage.setDriverId(profile.id);
          log.success("driver profile refreshed after retry");
          return;
        } catch (retryError) {
          if (isHttpAuthError(retryError)) {
            log.error("driver profile auth invalid after retry", { status: getHttpStatus(retryError) });
            await invokeForceLogoutDriver({
              reason: "profile_auth_invalid",
              severity: "AUTH_HARD_FAILURE",
              source: "driver",
              trigger_source: "manual_action",
            });
          } else {
            log.warn("driver profile retry failed", { error: retryError });
          }
        }
      } else if (isNetworkError(error)) {
        // Erreur réseau temporaire → ne pas déconnecter, juste logger
        log.warn("network error during profile refresh", { message: error?.message });
      } else {
        // Autres erreurs (500, etc.) → ne pas déconnecter non plus
        log.warn("server error during profile refresh", { status: status || "unknown" });
      }
    } finally {
      setDriverLoading(false);
    }
  }, [driverToken, forceLogoutDriverInternal]);

  const loginEnterpriseHandler = useCallback(
    async (params: EnterpriseLoginParams & { rememberMe?: boolean }) => {
      const { rememberMe, ...apiParams } = params;
      setEnterpriseLoading(true);
      try {
        const device = await ensureDeviceId();
        const response: EnterpriseLoginResponse = await loginEnterprise({
          ...apiParams,
          device_id: apiParams.device_id ?? device,
        });

        if ((response as EnterpriseLoginMfaPayload).mfa_required) {
          const mfa = response as EnterpriseLoginMfaPayload;
          const challenge: EnterpriseMfaChallenge = {
            challengeId: mfa.challenge_id,
            ttl: mfa.ttl,
            methods: mfa.methods ?? ["totp"],
            message: mfa.message,
          };
          setPendingEnterpriseMfa(challenge);
          if (rememberMe && params.email && params.password) {
            setPendingEnterpriseRememberMe({
              email: params.email.trim(),
              password: params.password,
            });
          }
          await storeMode("enterprise");
          return { mfaRequired: true as const, challenge };
        }

        await handleEnterpriseSuccess(response as EnterpriseTokenPayload);
        if (rememberMe && params.email && params.password) {
          try {
            await persistRememberMe(true, "enterprise");
            await setRememberedCredentials(
              params.email.trim(),
              params.password,
              "enterprise"
            );
          } catch {
            await persistRememberMe(false, "enterprise");
          }
        } else {
          await persistRememberMe(false, "enterprise");
          await clearRememberedCredentials("enterprise");
        }
        log.success("enterprise login succeeded, session stored");
        return { mfaRequired: false as const };
      } catch (error: any) {
        log.error("enterprise login failed", {
          message: error?.message,
          status: error?.response?.status,
          data: error?.response?.data,
          code: error?.code,
        });
        throw error;
      } finally {
        setEnterpriseLoading(false);
      }
    },
    [ensureDeviceId, handleEnterpriseSuccess, storeMode]
  );

  const verifyEnterpriseMfaHandler = useCallback(
    async (code: string, providedChallengeId?: string) => {
      const challengeId =
        providedChallengeId ?? pendingEnterpriseMfa?.challengeId;
      if (!challengeId) {
        throw new Error("Challenge MFA introuvable.");
      }
      setEnterpriseLoading(true);
      try {
        const device = await ensureDeviceId();
        const response = await verifyEnterpriseMfa({
          challenge_id: challengeId,
          code,
          device_id: device,
        });
        await handleEnterpriseSuccess(response);
        const pending = pendingEnterpriseRememberMe;
        setPendingEnterpriseRememberMe(null);
        if (pending) {
          try {
            await persistRememberMe(true, "enterprise");
            await setRememberedCredentials(
              pending.email,
              pending.password,
              "enterprise"
            );
          } catch {
            await persistRememberMe(false, "enterprise");
          }
        }
      } finally {
        setEnterpriseLoading(false);
      }
    },
    [
      ensureDeviceId,
      handleEnterpriseSuccess,
      pendingEnterpriseMfa,
      pendingEnterpriseRememberMe,
    ]
  );

  // ✅ Fonction pour charger la session depuis AsyncStorage sans faire de requête API
  // Utile après un switchMode quand la session vient d'être créée
  const loadEnterpriseSession = useCallback(async () => {

    setEnterpriseLoading(true);
    try {
      // ✅ CORRECTION : Utiliser SecureStore pour le token
      const [enterpriseToken, enterpriseSessionRaw] = await Promise.all([
        secureStorage.getEnterpriseToken(),
        AsyncStorage.getItem(ENTERPRISE_SESSION_KEY),
      ]);


      if (enterpriseToken && enterpriseSessionRaw) {
        const parsed: EnterpriseSessionState = JSON.parse(enterpriseSessionRaw);
        setEnterpriseSession({ ...parsed, token: enterpriseToken });


        log.success("enterprise session loaded from storage", {
          hasToken: !!enterpriseToken,
          hasRefreshToken: !!parsed.refreshToken,
          companyId: parsed.company?.id,
        });
      } else {
        setEnterpriseSession(null);
        log.info("no enterprise session in storage");
      }
    } catch (error) {
      log.error("enterprise session load error", { error });
      setEnterpriseSession(null);
    } finally {
      setEnterpriseLoading(false);
    }
  }, []);

  const refreshEnterprise = useCallback(async () => {
    // ✅ CORRECTION : Utiliser SecureStore pour le refresh token
    const refreshToken = await secureStorage.getEnterpriseRefreshToken();
    if (!refreshToken) return;
    try {
      const response = await runEnterpriseRefreshSingleflight(
        buildEnterpriseSessionKeyFromState(enterpriseSession),
        "manual_action",
        refreshToken
      );
      await handleEnterpriseSuccess(response);
    } catch (error) {
      const decision = shouldLogoutFromRefreshFailure(error, "refresh_endpoint");
      if (!decision.shouldLogout) {
        log.warn("enterprise refresh soft failure, keep session", {
          reason: decision.reason,
        });
        return;
      }
      const reason =
        getAuthFailureReason(error) === "refresh_expired"
          ? "refresh_expired"
          : "refresh_invalid";
      await invokeForceLogoutEnterprise({
        reason: reason as EnterpriseLogoutReason,
        severity: "AUTH_HARD_FAILURE",
        source: "enterprise",
        trigger_source: "manual_action",
      });
    }
  }, [handleEnterpriseSuccess]);

  const logoutEnterprise = useCallback(async () => {
    await invokeForceLogoutEnterprise({
      reason: "manual_logout",
      severity: "AUTH_MANUAL",
      source: "enterprise",
      trigger_source: "manual_action",
    });
  }, [forceLogoutEnterpriseInternal]);

  // P0 — Enregistrer les callbacks pour que les intercepteurs (api.ts, enterpriseAuth.ts) puissent invalider la session
  useEffect(() => {
    const unregDriver = registerForceLogoutDriver(forceLogoutDriverInternal);
    const unregEnterprise = registerForceLogoutEnterprise(forceLogoutEnterpriseInternal);
    return () => {
      unregDriver();
      unregEnterprise();
    };
  }, [forceLogoutDriverInternal, forceLogoutEnterpriseInternal]);

  // P2.2 — Alimenter log context (company_id, user_public_id_hash) pour corrélation multi-tenant
  useEffect(() => {
    if (driver?.user?.public_id) {
      setLogContextUser({ user_public_id: driver.user.public_id });
    }
    if (driver?.company?.id) {
      setLogContextUser({ company_id: driver.company.id });
    }
    if (enterpriseSession?.company?.id) {
      setLogContextUser({ company_id: enterpriseSession.company.id });
    }
    if (enterpriseSession?.user?.public_id) {
      setLogContextUser({ user_public_id: enterpriseSession.user.public_id });
    }
    if (deviceId) {
      setLogContextUser({ device_id: deviceId });
    }
  }, [driver?.user?.public_id, driver?.company?.id, enterpriseSession?.company?.id, enterpriseSession?.user?.public_id, deviceId]);

  const setMode = useCallback(
    async (nextMode: AuthMode) => {
      await storeMode(nextMode);
    },
    [storeMode]
  );

  const switchMode = useCallback(
    async (nextMode: AuthMode) => {
      await setMode(nextMode);
    },
    [setMode]
  );

  const isDriverAuthenticated = Boolean(driver && driverToken);
  const isEnterpriseAuthenticated = Boolean(enterpriseSession);
  const loading = initialLoading || driverLoading || enterpriseLoading;
  const isAuthenticated =
    mode === "enterprise" ? isEnterpriseAuthenticated : isDriverAuthenticated;

  const contextValue = useMemo<AuthContextType>(
    () => ({
      mode,
      setMode,
      switchMode,
      loading,
      deviceId,

      driver,
      token: driverToken,
      isDriverAuthenticated,
      driverLoading,
      login,
      logout,
      refreshProfile,

      enterpriseSession,
      isEnterpriseAuthenticated,
      enterpriseLoading,
      pendingEnterpriseMfa,
      loginEnterprise: loginEnterpriseHandler,
      verifyEnterpriseMfa: verifyEnterpriseMfaHandler,
      refreshEnterprise,
      loadEnterpriseSession,
      loadDriverSession,
      logoutEnterprise,

      isAuthenticated,
      authSessionState,
    }),
    [
      deviceId,
      driver,
      driverLoading,
      driverToken,
      enterpriseLoading,
      enterpriseSession,
      isAuthenticated,
      authSessionState,
      isDriverAuthenticated,
      isEnterpriseAuthenticated,
      loading,
      login,
      loginEnterpriseHandler,
      logout,
      logoutEnterprise,
      mode,
      pendingEnterpriseMfa,
      refreshEnterprise,
      refreshProfile,
      setMode,
      switchMode,
      verifyEnterpriseMfaHandler,
    ]
  );

  return (
    <AuthContext.Provider value={contextValue}>{children}</AuthContext.Provider>
  );
};

export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth doit être utilisé au sein d’un AuthProvider");
  }
  return context;
};
