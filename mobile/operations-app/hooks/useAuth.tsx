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
import { Alert, AppState, AppStateStatus } from "react-native";
import AsyncStorage from "@react-native-async-storage/async-storage";
import * as Crypto from "expo-crypto";

import {
  api,
  AuthResponse,
  Driver,
  fetchDriverProfile,
  loginDriver,
  refreshDriverTokenSingleflight,
  invalidateInterceptorCache,
} from "@/services/api";
import { secureStorage, asyncStorage } from "@/services/storage";
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
  refreshEnterpriseTokenSingleflight,
  refreshEnterpriseToken,
  verifyEnterpriseMfa,
  invalidateEnterpriseInterceptorCache,
} from "@/services/enterpriseAuth";
import { notifyAuthReady, notifyAuthNotReady } from "@/services/authSync";
import { setLogContextUser } from "@/services/logContext";
import {
  registerForceLogoutDriver,
  registerForceLogoutEnterprise,
  type DriverLogoutReason,
  type EnterpriseLogoutReason,
} from "@/services/authController";
import { isAuthNotReadyError } from "@/services/authGuards";
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
// ✅ PHASE 2 : Import de l'authentification biométrique
import {
  autoLoginWithBiometric,
  BiometricNoCredentialsError,
} from "@/services/biometricAuth";
import { sendIngestEvent } from "@/src/config/telemetry";
import { getLogger } from "@/utils/logger";

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
    params: EnterpriseLoginParams
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
  const [mode, setModeState] = useState<AuthMode>("enterprise");
  const [initialLoading, setInitialLoading] = useState(true);
  const [deviceId, setDeviceId] = useState<string | null>(null);

  const [driver, setDriver] = useState<Driver | null>(null);
  const [driverToken, setDriverToken] = useState<string | null>(null);
  const [driverLoading, setDriverLoading] = useState(false);

  const [enterpriseSession, setEnterpriseSession] =
    useState<EnterpriseSessionState | null>(null);
  const [enterpriseLoading, setEnterpriseLoading] = useState(false);
  const [pendingEnterpriseMfa, setPendingEnterpriseMfa] =
    useState<EnterpriseMfaChallenge | null>(null);

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
    async (reason: DriverLogoutReason) => {
      if (driverLogoutInProgressRef.current) return;
      driverLogoutInProgressRef.current = true;
      setDriverLoading(true);
      notifyAuthNotReady();
      try {
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
    async (reason: EnterpriseLogoutReason) => {
      if (enterpriseLogoutInProgressRef.current) return;
      enterpriseLogoutInProgressRef.current = true;
      try {
        if (shouldShowLogoutBanner(reason)) {
          await setLogoutMarker({ route: "enterprise", reason, ts: Date.now() });
        }
        notifyAuthNotReady();
        await clearEnterpriseStorage();
        invalidateEnterpriseInterceptorCache();
        setEnterpriseSession(null);
        setPendingEnterpriseMfa(null);
      } finally {
        enterpriseLogoutInProgressRef.current = false;
      }
    },
    [clearEnterpriseStorage]
  );

  const handleDriverLoginSuccess = useCallback(
    async (response: AuthResponse) => {
      pushSessionEvent("LOGIN_SUCCESS");
      // ✅ Les tokens sont déjà stockés dans loginDriver() (SecureStore + AsyncStorage)
      setDriverToken(response.token);
      pushSessionEvent("TOKEN_STORED");
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
          await forceLogoutDriverInternal("login_profile_failed");
        }
        throw error;
      }
    },
    [forceLogoutDriverInternal, storeMode]
  );

  const handleEnterpriseSuccess = useCallback(
    async (payload: EnterpriseTokenPayload) => {
      const session = parseEnterpriseSuccess(payload);
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
      await storeMode("enterprise");
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
    (async () => {
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
                const newAccessToken = await refreshDriverTokenSingleflight();
                setDriverToken(newAccessToken);
                notifyAuthReady();

                // S'assurer qu'on est en mode driver
                await storeMode("driver");

                // Charger le profil driver
                const profile = await fetchDriverProfile();
                if (isMounted) {
                  setDriver(profile);
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
                  await forceLogoutDriverInternal(
                    status === 401 ? "refresh_rejected_401" : "refresh_rejected_403"
                  );
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
                          "Reconnexion nécessaire",
                          "Vos identifiants enregistrés ne sont plus disponibles. Veuillez saisir vos identifiants pour vous reconnecter.",
                          [{ text: "Se connecter", style: "default" }]
                        );
                      }
                    } else {
                      log.warn("biometric auto-login failed", { error: autoLoginError });
                    }
                    // ✅ P0.2.B : Pas de forceLogout sur erreur réseau ou autre
                    if (isMounted && isHttpAuthError(autoLoginError)) {
                      const status = getHttpStatus(autoLoginError);
                      await forceLogoutDriverInternal(
                        status === 401 ? "refresh_rejected_401" : "refresh_rejected_403"
                      );
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
                      "Reconnexion nécessaire",
                      "Vos identifiants enregistrés ne sont plus disponibles. Veuillez saisir vos identifiants pour vous reconnecter.",
                      [{ text: "Se connecter", style: "default" }]
                    );
                  }
                } else {
                  log.warn("biometric auto-login failed", { error: autoLoginError });
                }
                // ✅ P0.2.B : Pas de forceLogout sur erreur réseau ; uniquement 401/403
                if (isMounted && isHttpAuthError(autoLoginError)) {
                  const status = getHttpStatus(autoLoginError);
                  await forceLogoutDriverInternal(
                    status === 401 ? "refresh_rejected_401" : "refresh_rejected_403"
                  );
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
                      const refreshResponse =
                        await refreshEnterpriseTokenSingleflight(refreshToken);
                      await handleEnterpriseSuccess(refreshResponse);
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
            await clearEnterpriseStorage();
            setEnterpriseSession(null);
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
    })();
    return () => {
      isMounted = false;
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

            const newAccessToken = await refreshDriverTokenSingleflight();
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
            await forceLogoutDriverInternal("refresh_rejected_403");
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
            const newAccessToken = await refreshDriverTokenSingleflight();
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
            await forceLogoutDriverInternal("refresh_rejected_403");
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
        let resyncOk = true;
        try {
          // 1) Recharger tokens depuis storage et refresh si expiré / proche expiration
          const expiresAt = getTokenExpiration(driverToken);
          if (!expiresAt) {
            const refreshToken = await secureStorage.getRefreshToken();
            if (refreshToken) {
              try {
                const newAccessToken = await refreshDriverTokenSingleflight();
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
                  const newAccessToken = await refreshDriverTokenSingleflight();
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
      } else {
        pushSessionEvent("APP_BACKGROUND");
      }
    };

    const subscription = AppState.addEventListener("change", handleAppStateChange);
    return () => subscription.remove();
  }, [driverToken, mode]);

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

            const refreshResponse =
              await refreshEnterpriseTokenSingleflight(refreshToken);
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
              await forceLogoutEnterpriseInternal("refresh_rejected_403");
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
            const refreshResponse =
              await refreshEnterpriseTokenSingleflight(refreshToken);
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
            await forceLogoutEnterpriseInternal("refresh_rejected_403");
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
    // #region agent log
    debugLog({ location: 'useAuth.tsx:useEffect:switchMode', message: 'useEffect switchMode entry', data: { mode, hasEnterpriseSession: !!enterpriseSession, initialLoading }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'G' });
    // #endregion

    if (mode !== "enterprise" || enterpriseSession || initialLoading) {
      // #region agent log
      if (mode === "enterprise") {
        debugLog({ location: 'useAuth.tsx:useEffect:switchMode', message: 'useEffect switchMode skipped', data: { reason: enterpriseSession ? 'hasSession' : initialLoading ? 'loading' : 'notEnterprise' }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'G' });
      }
      // #endregion
      return;
    }

    let isMounted = true;
    (async () => {
      try {
        // #region agent log
        debugLog({ location: 'useAuth.tsx:useEffect:switchMode', message: 'useEffect switchMode loading session', data: {}, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'G' });
        // #endregion

        // ✅ CORRECTION : Utiliser SecureStore pour le token
        const [enterpriseToken, enterpriseSessionRaw, justCreated] = await Promise.all([
          secureStorage.getEnterpriseToken(),
          AsyncStorage.getItem(ENTERPRISE_SESSION_KEY),
          AsyncStorage.getItem("enterprise_session_just_created"),
        ]);

        // #region agent log
        debugLog({ location: 'useAuth.tsx:useEffect:switchMode', message: 'useEffect switchMode session loaded', data: { hasToken: !!enterpriseToken, hasSession: !!enterpriseSessionRaw, justCreated }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'G' });
        // #endregion

        if (enterpriseToken && enterpriseSessionRaw && isMounted) {
          try {
            const parsed: EnterpriseSessionState = JSON.parse(enterpriseSessionRaw);
            // Restaurer la session depuis le stockage
            setEnterpriseSession({ ...parsed, token: enterpriseToken });

            // #region agent log
            debugLog({ location: 'useAuth.tsx:useEffect:switchMode', message: 'useEffect switchMode session set', data: { hasToken: !!enterpriseToken, companyId: parsed.company?.id }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'G' });
            // #endregion

            // Si la session vient d'être créée (juste après un switch), ne pas la vérifier immédiatement
            if (justCreated === "true") {
              await AsyncStorage.removeItem("enterprise_session_just_created");
              log.info("enterprise session loaded after switch, verification deferred");
            }
          } catch (error) {
            log.warn("enterprise session load error after switch", { error });
            // #region agent log
            debugLog({ location: 'useAuth.tsx:useEffect:switchMode', message: 'useEffect switchMode error', data: { error: String(error) }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'G' });
            // #endregion
          }
        }
      } catch (error) {
        log.warn("enterprise async storage read error after switch", { error });
        // #region agent log
        debugLog({ location: 'useAuth.tsx:useEffect:switchMode', message: 'useEffect switchMode AsyncStorage error', data: { error: String(error) }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'G' });
        // #endregion
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
    await forceLogoutDriverInternal("manual_logout");
  }, [forceLogoutDriverInternal]);

  // ✅ Fonction pour charger la session driver depuis SecureStorage sans faire de requête API
  // Utile après un switchMode quand la session vient d'être créée
  const loadDriverSession = useCallback(async () => {
    // #region agent log
    sendIngestEvent({ location: 'useAuth.tsx:loadDriverSession', message: 'loadDriverSession entry', data: {}, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'K' });
    // #endregion

    setDriverLoading(true);
    try {
      const [accessToken, refreshToken, userPublicId] = await Promise.all([
        secureStorage.getAccessToken(),
        secureStorage.getRefreshToken(),
        secureStorage.getUserPublicId(),
      ]);

      // #region agent log
      sendIngestEvent({ location: 'useAuth.tsx:loadDriverSession', message: 'loadDriverSession loaded', data: { hasToken: !!accessToken, hasRefreshToken: !!refreshToken, hasUserPublicId: !!userPublicId }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'K' });
      // #endregion

      if (accessToken) {
        setDriverToken(accessToken);

        // Charger le profil driver pour mettre à jour le contexte
        try {
          const profile = await fetchDriverProfile();
          setDriver(profile);
          await asyncStorage.setDriverId(profile.id);

          // #region agent log
          debugLog({ location: 'useAuth.tsx:loadDriverSession', message: 'loadDriverSession set', data: { hasToken: !!accessToken, driverId: profile.id }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'K' });
          // #endregion

          log.success("driver session loaded from storage", {
            hasToken: !!accessToken,
            hasRefreshToken: !!refreshToken,
            driverId: profile.id,
          });
        } catch (profileError) {
          log.warn("driver profile load error", { error: profileError });
          // #region agent log
          debugLog({ location: 'useAuth.tsx:loadDriverSession', message: 'loadDriverSession profile error', data: { error: String(profileError) }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'K' });
          // #endregion
          // Ne pas nettoyer la session ici, le token est valide mais le profil ne peut pas être chargé
        }
      } else {
        setDriver(null);
        setDriverToken(null);
        log.info("no driver session in storage");
      }
    } catch (error) {
      log.error("driver session load error", { error });
      // #region agent log
      sendIngestEvent({ location: 'useAuth.tsx:loadDriverSession', message: 'loadDriverSession error', data: { error: String(error) }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'K' });
      // #endregion
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
      await asyncStorage.setDriverId(profile.id);
      log.success("driver profile refreshed", { profileId: profile.id });
    } catch (error: any) {
      const status = error?.response?.status;

      // P1.B: Sur 401/403, tenter refresh+retry avant logout. Réseau/5xx => jamais logout.
      if (isHttpAuthError(error)) {
        try {
          const newToken = await refreshDriverTokenSingleflight();
          setDriverToken(newToken);
          invalidateInterceptorCache();
          const profile = await fetchDriverProfile();
          setDriver(profile);
          await asyncStorage.setDriverId(profile.id);
          log.success("driver profile refreshed after retry");
          return;
        } catch (retryError) {
          if (isHttpAuthError(retryError)) {
            log.error("driver profile auth invalid after retry", { status: getHttpStatus(retryError) });
            await forceLogoutDriverInternal("profile_auth_invalid");
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
    async (params: EnterpriseLoginParams) => {
      setEnterpriseLoading(true);
      try {
        const device = await ensureDeviceId();
        const response: EnterpriseLoginResponse = await loginEnterprise({
          ...params,
          device_id: params.device_id ?? device,
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
          await storeMode("enterprise");
          return { mfaRequired: true as const, challenge };
        }

        await handleEnterpriseSuccess(response as EnterpriseTokenPayload);
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
      } finally {
        setEnterpriseLoading(false);
      }
    },
    [ensureDeviceId, handleEnterpriseSuccess, pendingEnterpriseMfa]
  );

  // ✅ Fonction pour charger la session depuis AsyncStorage sans faire de requête API
  // Utile après un switchMode quand la session vient d'être créée
  const loadEnterpriseSession = useCallback(async () => {
    // #region agent log
    sendIngestEvent({ location: 'useAuth.tsx:loadEnterpriseSession', message: 'loadEnterpriseSession entry', data: {}, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'H' });
    // #endregion

    setEnterpriseLoading(true);
    try {
      // ✅ CORRECTION : Utiliser SecureStore pour le token
      const [enterpriseToken, enterpriseSessionRaw] = await Promise.all([
        secureStorage.getEnterpriseToken(),
        AsyncStorage.getItem(ENTERPRISE_SESSION_KEY),
      ]);

      // #region agent log
      sendIngestEvent({ location: 'useAuth.tsx:loadEnterpriseSession', message: 'loadEnterpriseSession loaded', data: { hasToken: !!enterpriseToken, hasSession: !!enterpriseSessionRaw }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'H' });
      // #endregion

      if (enterpriseToken && enterpriseSessionRaw) {
        const parsed: EnterpriseSessionState = JSON.parse(enterpriseSessionRaw);
        setEnterpriseSession({ ...parsed, token: enterpriseToken });

        // #region agent log
        debugLog({ location: 'useAuth.tsx:loadEnterpriseSession', message: 'loadEnterpriseSession set', data: { hasToken: !!enterpriseToken, companyId: parsed.company?.id }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'H' });
        // #endregion

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
      // #region agent log
      sendIngestEvent({ location: 'useAuth.tsx:loadEnterpriseSession', message: 'loadEnterpriseSession error', data: { error: String(error) }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'H' });
      // #endregion
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
      const response = await refreshEnterpriseTokenSingleflight(refreshToken);
      await handleEnterpriseSuccess(response);
    } catch (error) {
      log.warn("enterprise refresh token invalid", { error });
      await forceLogoutEnterpriseInternal("refresh_rejected_401");
    }
  }, [handleEnterpriseSuccess, forceLogoutEnterpriseInternal]);

  const logoutEnterprise = useCallback(async () => {
    await forceLogoutEnterpriseInternal("manual_logout");
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
    }),
    [
      deviceId,
      driver,
      driverLoading,
      driverToken,
      enterpriseLoading,
      enterpriseSession,
      isAuthenticated,
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
