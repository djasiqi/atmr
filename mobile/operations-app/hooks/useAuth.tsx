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
  refreshAccessToken,
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
import { setLogoutMarker, isSessionExpiredReason } from "@/services/logoutMarker";
import { connectSocket, disconnectSocket } from "@/services/socket";
// ✅ PHASE 2 : Import de l'authentification biométrique
import {
  autoLoginWithBiometric,
  BiometricNoCredentialsError,
} from "@/services/biometricAuth";
import { sendIngestEvent } from "@/src/config/telemetry";

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
    console.warn("[getTokenExpiration] Erreur décodage JWT:", error);
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
        if (isSessionExpiredReason(reason)) {
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
              console.warn("Erreur lors du logout server-side:", e);
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
        if (isSessionExpiredReason(reason)) {
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
        console.warn("Impossible de récupérer le profil chauffeur :", error);
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
        // eslint-disable-next-line no-console
        console.error("[ENT] ERREUR: Token manquant dans la réponse de login");
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
        // eslint-disable-next-line no-console
        console.error("[ENT] ERREUR: Token stocké ne correspond pas au token reçu");
      }

      // Attendre un peu pour s'assurer que AsyncStorage a bien écrit les données
      // avant que d'autres requêtes ne soient faites
      await new Promise(resolve => setTimeout(resolve, 100));

      // eslint-disable-next-line no-console
      console.log("[ENT] Session entreprise stockée avec succès", {
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
                console.warn("Access token chauffeur invalide, tentative avec refresh token :", error);
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
                console.warn(
                  "Auto-login échoué (refresh token) :",
                  refreshError
                );

                // ✅ P0.2.B : Offline/timeout/5xx ≠ logout — uniquement 401/403 invalide la session
                if (isNetworkError(refreshError)) {
                  console.warn("[useAuth] ⚠️ Boot: erreur réseau, tokens conservés. Connexion requise.");
                  // Pas de forceLogout, pas de wipe storage
                } else if (isHttpAuthError(refreshError)) {
                  const status = getHttpStatus(refreshError);
                  await forceLogoutDriverInternal(
                    status === 401 ? "refresh_rejected_401" : "refresh_rejected_403"
                  );
                } else {
                  // 5xx, autre → pas de logout
                  console.warn("[useAuth] ⚠️ Boot: erreur serveur/autre, tokens conservés.");
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
                        console.log("[useAuth] ✅ Auto-login réussi avec authentification biométrique");
                      }
                    } else {
                      if (__DEV__) {
                        console.log("[useAuth] ⚠️ Auto-login biométrique annulé ou échoué");
                      }
                    }
                  } catch (autoLoginError) {
                    if (autoLoginError instanceof BiometricNoCredentialsError) {
                      if (isMounted) {
                        Alert.alert(
                          "",
                          "Identifiants mémorisés indisponibles. Veuillez vous reconnecter."
                        );
                      }
                    } else {
                      console.warn("[useAuth] ⚠️ Auto-login biométrique échoué:", autoLoginError);
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
                    console.log("[useAuth] ✅ Auto-login réussi avec authentification biométrique");
                  }
                } else {
                  // ✅ P0.2.B : User annulé → pas de forceLogout (rien à effacer)
                }
              } catch (autoLoginError) {
                if (autoLoginError instanceof BiometricNoCredentialsError) {
                  if (isMounted) {
                    Alert.alert(
                      "",
                      "Identifiants mémorisés indisponibles. Veuillez vous reconnecter."
                    );
                  }
                } else {
                  console.warn("[useAuth] ⚠️ Auto-login biométrique échoué:", autoLoginError);
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
              // eslint-disable-next-line no-console
              console.log("[ENT] Session vient d'être créée, vérification différée");
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
                      // eslint-disable-next-line no-console
                      console.warn("Rafraîchissement token entreprise échoué :", refreshError);
                      // Ne pas nettoyer la session ici - laisser l'utilisateur utiliser l'app
                      // La session sera invalidée lors de la prochaine requête API réelle
                    }
                  } else {
                    // eslint-disable-next-line no-console
                    console.warn("Session entreprise invalide (pas de refresh token) :", sessionError);
                    // Ne pas nettoyer immédiatement - laisser l'utilisateur utiliser l'app
                  }
                } else {
                  // eslint-disable-next-line no-console
                  console.warn("Erreur lors de la vérification de session entreprise :", sessionError);
                  // Ne pas nettoyer la session pour les erreurs réseau temporaires
                }
              }
            })();
          } catch (error) {
            // eslint-disable-next-line no-console
            console.warn("Erreur lors de la restauration de la session entreprise :", error);
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
      console.warn("[useAuth] Impossible de décoder l'expiration du token driver");
      return;
    }

    const now = Date.now();
    const timeUntilExpiry = expiresAt - now;
    const refreshBeforeExpiry = 10 * 60 * 1000; // 10 minutes

    const runProactiveRefresh = () => {
      // ✅ P0.3.A : Vérifier cooldown avant tentative
      if (isDriverProactiveRefreshInCooldown()) {
        const remaining = getDriverProactiveRefreshCooldownRemaining();
        console.warn(`[useAuth] ⏳ Refresh proactif driver en cooldown, retry dans ${Math.round(remaining / 1000)}s`);
        driverProactiveRefreshTimeoutRef.current = setTimeout(runProactiveRefresh, remaining);
        return;
      }

      (async () => {
        console.log("[useAuth] 🔄 Refresh proactif du token driver (10min avant expiration)");

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

            console.log(`[useAuth] ✅ Refresh proactif réussi${retryCount > 0 ? ` (après ${retryCount} tentative(s))` : ""}`);
            return;
          } catch (error: any) {
            lastError = error;
            const status = error?.response?.status;
            const isNetworkError = !error?.response; // Pas de réponse = erreur réseau

            // ✅ PHASE 3 : Ne pas retry pour les erreurs critiques (401, 403)
            if (status === 401 || status === 403) {
              console.error(`[useAuth] ❌ Refresh proactif échoué (${status}):`, error?.response?.data || error?.message);
              break; // Sortir de la boucle, ne pas retry
            }

            // ✅ PHASE 3 : Retry uniquement pour erreurs réseau ou serveur (500, etc.)
            if (isNetworkError || (status && status >= 500)) {
              retryCount++;
              if (retryCount < MAX_RETRIES) {
                const delay = Math.min(1000 * Math.pow(2, retryCount - 1), 10000); // Backoff exponentiel (max 10s)
                console.warn(`[useAuth] ⚠️ Refresh proactif échoué (tentative ${retryCount}/${MAX_RETRIES}), retry dans ${delay}ms...`);
                await new Promise(resolve => setTimeout(resolve, delay));
                continue; // Retry
              } else {
                console.error(`[useAuth] ❌ Refresh proactif échoué après ${MAX_RETRIES} tentatives`);
                break; // Toutes les tentatives ont échoué
              }
            } else {
              // Autres erreurs (400, etc.) → ne pas retry
              console.warn(`[useAuth] ⚠️ Refresh proactif échoué (status: ${status}), pas de retry`);
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
            console.warn(
              "[useAuth] ⚠️ Refresh proactif échoué (status=%s) mais access token encore valide. Pas de logout.",
              status
            );
            const backoff = recordDriverProactiveRefreshFailure();
            driverProactiveRefreshTimeoutRef.current = setTimeout(runProactiveRefresh, backoff);
            return;
          }

          // ✅ Si 403 (compte désactivé) et access expiré, forcer déconnexion
          if (status === 403) {
            console.error(
              "[useAuth] 🚫 Compte désactivé (403) lors du refresh proactif. Déconnexion forcée.",
              errorData
            );
            await forceLogoutDriverInternal("refresh_rejected_403");
            return;
          }

          // 401 et access expiré, ou 5xx/network → cooldown + retry
          const backoff = recordDriverProactiveRefreshFailure();
          driverProactiveRefreshTimeoutRef.current = setTimeout(runProactiveRefresh, backoff);
          console.warn("[useAuth] ⚠️ Refresh proactif échoué (fallback sur intercepteur 401):", lastError);
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

      console.log(`[useAuth] ⏰ Refresh proactif planifié dans ${Math.round(initialDelay / 1000 / 60)} minutes`);

      return () => {
        if (driverProactiveRefreshTimeoutRef.current) {
          clearTimeout(driverProactiveRefreshTimeoutRef.current);
          driverProactiveRefreshTimeoutRef.current = null;
        }
      };
    } else if (timeUntilExpiry > 0 && timeUntilExpiry <= refreshBeforeExpiry) {
      // ✅ PHASE 3 : Token expire bientôt (< 10min), rafraîchir immédiatement avec retry
      console.log("[useAuth] ⚡ Token expire dans moins de 10min, refresh immédiat");
      (async () => {
        const MAX_RETRIES = 3;
        let retryCount = 0;
        let lastError: any = null;

        while (retryCount < MAX_RETRIES) {
          try {
            const refreshToken = await secureStorage.getRefreshToken();
            if (!refreshToken) {
              throw new Error("Pas de refresh token disponible");
            }

            const refreshResponse = await refreshAccessToken(refreshToken);
            await secureStorage.setAccessToken(refreshResponse.access_token);
            setDriverToken(refreshResponse.access_token);
            if (refreshResponse.refresh_token) {
              await secureStorage.setRefreshToken(refreshResponse.refresh_token);
            }
            invalidateInterceptorCache();
            resetDriverProactiveRefreshCooldown();
            console.log(`[useAuth] ✅ Refresh immédiat réussi${retryCount > 0 ? ` (après ${retryCount} tentative(s))` : ""}`);
            return;
          } catch (error: any) {
            lastError = error;
            const status = error?.response?.status;
            const isNetworkError = !error?.response;

            // Ne pas retry pour erreurs critiques
            if (status === 401 || status === 403) {
              console.error(`[useAuth] ❌ Refresh immédiat échoué (${status})`);
              break;
            }

            // Retry pour erreurs réseau ou serveur
            if (isNetworkError || (status && status >= 500)) {
              retryCount++;
              if (retryCount < MAX_RETRIES) {
                const delay = Math.min(1000 * Math.pow(2, retryCount - 1), 10000);
                console.warn(`[useAuth] ⚠️ Refresh immédiat échoué (tentative ${retryCount}/${MAX_RETRIES}), retry dans ${delay}ms...`);
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
            console.warn(
              "[useAuth] ⚠️ Refresh immédiat échoué (status=%s) mais access encore valide. Pas de logout.",
              status
            );
            recordDriverProactiveRefreshFailure();
            return;
          }

          if (status === 403) {
            console.error("[useAuth] 🚫 Compte désactivé (403) lors du refresh immédiat. Déconnexion forcée.", errorData);
            await forceLogoutDriverInternal("refresh_rejected_403");
            return;
          }

          recordDriverProactiveRefreshFailure();
          console.warn("[useAuth] ⚠️ Refresh immédiat échoué:", lastError);
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
                  console.warn("[useAuth] FOREGROUND_RESYNC ping /driver/me échoué après 2 essais");
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
      console.warn("[useAuth] Impossible de décoder l'expiration du token entreprise");
      return;
    }

    const now = Date.now();
    const timeUntilExpiry = expiresAt - now;
    const refreshBeforeExpiry = 5 * 60 * 1000; // 5 minutes

    if (timeUntilExpiry > refreshBeforeExpiry) {
      const runEnterpriseProactiveRefresh = () => {
        if (isEnterpriseProactiveRefreshInCooldown()) {
          const remaining = getEnterpriseProactiveRefreshCooldownRemaining();
          console.warn(`[useAuth] ⏳ Refresh proactif entreprise en cooldown, retry dans ${Math.round(remaining / 1000)}s`);
          enterpriseProactiveRefreshTimeoutRef.current = setTimeout(runEnterpriseProactiveRefresh, remaining);
          return;
        }

        (async () => {
          console.log("[useAuth] 🔄 Refresh proactif du token entreprise (5min avant expiration)");
          try {
            // ✅ CORRECTION : Utiliser SecureStore pour le refresh token
            const refreshToken = enterpriseSession.refreshToken || await secureStorage.getEnterpriseRefreshToken();
            if (!refreshToken) {
              console.warn("[useAuth] ⚠️ Aucun refresh token disponible pour le refresh proactif entreprise");
              return;
            }

            if (__DEV__) {
              console.log("[useAuth] 🔄 Refresh proactif entreprise: token disponible (longueur:", refreshToken.length, ")");
            }

            const refreshResponse =
              await refreshEnterpriseTokenSingleflight(refreshToken);
            await handleEnterpriseSuccess(refreshResponse);

            // ⚡ CORRECTION : Invalider le cache interceptor pour forcer l'utilisation du nouveau token
            invalidateEnterpriseInterceptorCache();

            console.log("[useAuth] ✅ Refresh proactif entreprise réussi");
          } catch (error: any) {
            const status = error?.response?.status;
            const errorData = error?.response?.data;
            const errorMessage = errorData?.error || error?.message;

            // ✅ Si 401 (refresh token invalide/expiré), ne pas déconnecter immédiatement
            // L'intercepteur gérera la déconnexion lors de la prochaine requête
            if (status === 401) {
              console.error(
                `[useAuth] ❌ Refresh token invalide (401) lors du refresh proactif entreprise: ${errorMessage}`
              );
              // Ne pas déconnecter ici, l'intercepteur gérera lors de la prochaine requête réelle
              return;
            }

            // ✅ P0.2.A : 403 mais access encore valide → best effort, pas de logout
            if (status === 403 && enterpriseSession && accessStillValid(enterpriseSession.token)) {
              console.warn(
                "[useAuth] ⚠️ Refresh proactif entreprise échoué (403) mais access encore valide. Pas de logout."
              );
              const backoff = recordEnterpriseProactiveRefreshFailure();
              enterpriseProactiveRefreshTimeoutRef.current = setTimeout(runEnterpriseProactiveRefresh, backoff);
              return;
            }

            if (status === 403) {
              console.error(
                "[useAuth] 🚫 Compte désactivé (403) lors du refresh proactif entreprise. Déconnexion forcée.",
                errorData
              );
              await forceLogoutEnterpriseInternal("refresh_rejected_403");
              return;
            }

            // Autres erreurs (réseau, serveur, etc.) → cooldown + retry
            const backoff = recordEnterpriseProactiveRefreshFailure();
            enterpriseProactiveRefreshTimeoutRef.current = setTimeout(runEnterpriseProactiveRefresh, backoff);
            console.warn(
              `[useAuth] ⚠️ Refresh proactif entreprise échoué (status: ${status || "network"}): ${errorMessage}`
            );
          }
        })();
      };

      const initialDelay = isEnterpriseProactiveRefreshInCooldown()
        ? getEnterpriseProactiveRefreshCooldownRemaining()
        : timeUntilExpiry - refreshBeforeExpiry;
      const timeoutId = setTimeout(runEnterpriseProactiveRefresh, initialDelay);
      enterpriseProactiveRefreshTimeoutRef.current = timeoutId;

      console.log(`[useAuth] ⏰ Refresh proactif entreprise planifié dans ${Math.round(initialDelay / 1000 / 60)} minutes`);

      return () => {
        if (enterpriseProactiveRefreshTimeoutRef.current) {
          clearTimeout(enterpriseProactiveRefreshTimeoutRef.current);
          enterpriseProactiveRefreshTimeoutRef.current = null;
        }
      };
    } else if (timeUntilExpiry > 0 && timeUntilExpiry <= refreshBeforeExpiry) {
      // Token expire bientôt (< 5min), rafraîchir immédiatement
      console.log("[useAuth] ⚡ Token entreprise expire dans moins de 5min, refresh immédiat");
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

            console.log("[useAuth] ✅ Refresh immédiat entreprise réussi");
          }
        } catch (error: any) {
          const status = error?.response?.status;
          const errorData = error?.response?.data;

          // ✅ P0.2.A : 403 mais access encore valide → pas de logout
          if (status === 403 && enterpriseSession && accessStillValid(enterpriseSession.token)) {
            console.warn(
              "[useAuth] ⚠️ Refresh immédiat entreprise échoué (403) mais access encore valide. Pas de logout."
            );
            recordEnterpriseProactiveRefreshFailure();
            return;
          }

          if (status === 403) {
            console.error(
              "[useAuth] 🚫 Compte désactivé (403) lors du refresh immédiat entreprise. Déconnexion forcée.",
              errorData
            );
            await forceLogoutEnterpriseInternal("refresh_rejected_403");
            return;
          }

          recordEnterpriseProactiveRefreshFailure();
          console.warn("[useAuth] ⚠️ Refresh immédiat entreprise échoué:", error);
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
              // eslint-disable-next-line no-console
              console.log("[ENT] Session chargée après switchMode, vérification différée");
            }
          } catch (error) {
            // eslint-disable-next-line no-console
            console.warn("[ENT] Erreur lors du chargement de la session après switchMode:", error);
            // #region agent log
            debugLog({ location: 'useAuth.tsx:useEffect:switchMode', message: 'useEffect switchMode error', data: { error: String(error) }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'G' });
            // #endregion
          }
        }
      } catch (error) {
        // eslint-disable-next-line no-console
        console.warn("[ENT] Erreur lors de la lecture AsyncStorage après switchMode:", error);
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

          console.log("[useAuth] Driver session loaded from SecureStorage:", {
            hasToken: !!accessToken,
            hasRefreshToken: !!refreshToken,
            driverId: profile.id,
          });
        } catch (profileError) {
          console.warn("[useAuth] Error loading driver profile:", profileError);
          // #region agent log
          debugLog({ location: 'useAuth.tsx:loadDriverSession', message: 'loadDriverSession profile error', data: { error: String(profileError) }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'K' });
          // #endregion
          // Ne pas nettoyer la session ici, le token est valide mais le profil ne peut pas être chargé
        }
      } else {
        setDriver(null);
        setDriverToken(null);
        console.log("[useAuth] No driver session found in SecureStorage.");
      }
    } catch (error) {
      console.error("[useAuth] Error loading driver session:", error);
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
        console.log("[useAuth] Token driver récupéré depuis SecureStorage pour refreshProfile");
      } else {
        console.warn("[useAuth] Aucun token driver disponible pour rafraîchir le profil");
        return;
      }
    }

    setDriverLoading(true);
    try {
      const profile = await fetchDriverProfile();
      setDriver(profile);
      await asyncStorage.setDriverId(profile.id);
      console.log("[useAuth] Profil chauffeur rafraîchi et stocké:", profile.id);
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
          console.log("[useAuth] Profil rafraîchi après retry post-refresh");
          return;
        } catch (retryError) {
          if (isHttpAuthError(retryError)) {
            console.error(
              "[useAuth] ❌ Profile 401/403 persistant après refresh+retry. Déconnexion.",
              getHttpStatus(retryError)
            );
            await forceLogoutDriverInternal("profile_auth_invalid");
          } else {
            console.warn("[useAuth] Retry profile échoué (non-auth):", retryError);
          }
        }
      } else if (isNetworkError(error)) {
        // Erreur réseau temporaire → ne pas déconnecter, juste logger
        console.warn(
          "[useAuth] ⚠️ Erreur réseau lors du refresh profil (pas de connexion). Profil non mis à jour mais utilisateur reste connecté.",
          error?.message
        );
      } else {
        // Autres erreurs (500, etc.) → ne pas déconnecter non plus
        console.warn(
          "[useAuth] ⚠️ Erreur serveur lors du refresh profil (status: %s). Profil non mis à jour mais utilisateur reste connecté.",
          status || "unknown"
        );
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
        // eslint-disable-next-line no-console
        console.log("[ENT] Login réussi, session stockée");
        return { mfaRequired: false as const };
      } catch (error: any) {
        // eslint-disable-next-line no-console
        console.error("[ENT] Erreur lors du login:", {
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

        console.log("[useAuth] Enterprise session loaded from AsyncStorage:", {
          hasToken: !!enterpriseToken,
          hasRefreshToken: !!parsed.refreshToken,
          companyId: parsed.company?.id,
        });
      } else {
        setEnterpriseSession(null);
        console.log("[useAuth] No enterprise session found in AsyncStorage.");
      }
    } catch (error) {
      console.error("[useAuth] Error loading enterprise session:", error);
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
      console.warn("Refresh token entreprise invalide :", error);
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
