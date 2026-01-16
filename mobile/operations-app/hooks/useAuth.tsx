import React, {
  createContext,
  ReactNode,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import AsyncStorage from "@react-native-async-storage/async-storage";
import * as Crypto from "expo-crypto";

import {
  api,
  AuthResponse,
  Driver,
  fetchDriverProfile,
  loginDriver,
  refreshAccessToken,
  invalidateInterceptorCache,
} from "@/services/api";
import { secureStorage, asyncStorage } from "@/services/storage";
import {
  ENTERPRISE_REFRESH_KEY,
  ENTERPRISE_SESSION_KEY,
  ENTERPRISE_TOKEN_KEY,
  EnterpriseLoginParams,
  EnterpriseLoginResponse,
  EnterpriseLoginMfaPayload,
  EnterpriseTokenPayload,
  fetchEnterpriseSession,
  loginEnterprise,
  refreshEnterpriseToken,
  verifyEnterpriseMfa,
} from "@/services/enterpriseAuth";

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
  login: (email: string, password: string) => Promise<void>;
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

  const clearDriverStorage = useCallback(async () => {
    await secureStorage.clearAll();
    await asyncStorage.clearAuth();
  }, []);

  const clearEnterpriseStorage = useCallback(async () => {
    await AsyncStorage.multiRemove([
      ENTERPRISE_TOKEN_KEY,
      ENTERPRISE_REFRESH_KEY,
      ENTERPRISE_SESSION_KEY,
    ]);
  }, []);

  const handleDriverLoginSuccess = useCallback(
    async (response: AuthResponse) => {
      // ✅ Les tokens sont déjà stockés dans loginDriver() (SecureStore + AsyncStorage)
      setDriverToken(response.token);
      await storeMode("driver");
      try {
        const profile = await fetchDriverProfile();
        setDriver(profile);
        // ✅ Stocker driver_id pour navigation rapide
        await asyncStorage.setDriverId(profile.id);
      } catch (error) {
        console.warn("Impossible de récupérer le profil chauffeur :", error);
        await clearDriverStorage();
        setDriver(null);
        setDriverToken(null);
        throw error;
      }
    },
    [clearDriverStorage, storeMode]
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

      // Stocker le token et la session de manière synchrone pour éviter les problèmes de timing
      await AsyncStorage.multiSet([
        [ENTERPRISE_TOKEN_KEY, session.token],
        [ENTERPRISE_SESSION_KEY, JSON.stringify(session)],
        // Marquer que la session vient d'être créée pour éviter la vérification immédiate
        ["enterprise_session_just_created", "true"],
      ]);
      if (session.refreshToken) {
        await AsyncStorage.setItem(
          ENTERPRISE_REFRESH_KEY,
          session.refreshToken
        );
      } else {
        await AsyncStorage.removeItem(ENTERPRISE_REFRESH_KEY);
      }
      await storeMode("enterprise");

      // Vérifier que le token a bien été stocké
      const storedToken = await AsyncStorage.getItem(ENTERPRISE_TOKEN_KEY);
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
    (async () => {
      try {
        // ⚡ OPTIMISATION Phase 2 : Lecture parallèle des tokens et données de stockage
        // Réduit le temps de démarrage de ~20-30ms à ~10-15ms
        const [storedMode, storedDevice, refreshToken, accessToken] =
          await Promise.all([
            AsyncStorage.getItem(MODE_KEY),
            AsyncStorage.getItem(ENTERPRISE_DEVICE_KEY),
            secureStorage.getRefreshToken(),
            secureStorage.getAccessToken(),
          ]);

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
            // ⚡ OPTIMISATION : accessToken déjà lu en parallèle ci-dessus
            if (accessToken) {
              try {
                setDriverToken(accessToken);
                const profile = await fetchDriverProfile();
                if (isMounted) {
                  setDriver(profile);
                  await asyncStorage.setDriverId(profile.id);
                  profileLoaded = true;
                  // S'assurer qu'on est en mode driver
                  await storeMode("driver");
                }
              } catch (error) {
                console.warn("Access token chauffeur invalide, tentative avec refresh token :", error);
                // Continuer avec le refresh token si l'access token échoue
              }
            }

            // 2. Si l'access token n'a pas fonctionné, essayer le refresh token
            if (!profileLoaded && refreshToken) {
              try {
                const refreshResponse = await refreshAccessToken(refreshToken);

                // Stocker le nouveau access_token dans SecureStore
                if (refreshResponse.access_token) {
                  await secureStorage.setAccessToken(refreshResponse.access_token);
                  setDriverToken(refreshResponse.access_token);
                }

                // Mettre à jour refresh_token si rotation activée
                if (refreshResponse.refresh_token) {
                  await secureStorage.setRefreshToken(refreshResponse.refresh_token);
                }

                // S'assurer qu'on est en mode driver
                await storeMode("driver");

                // Charger le profil driver
                const profile = await fetchDriverProfile();
                if (isMounted) {
                  setDriver(profile);
                  await asyncStorage.setDriverId(profile.id);
                  profileLoaded = true;
                }
              } catch (refreshError) {
                console.warn(
                  "Auto-login échoué (refresh token invalide) :",
                  refreshError
                );
                // Nettoyer seulement si on n'a pas réussi à charger le profil
                if (!profileLoaded) {
                  await secureStorage.clearAll();
                  await asyncStorage.clearAuth();
                  if (isMounted) {
                    setDriver(null);
                    setDriverToken(null);
                  }
                }
              }
            }

            // 3. Si aucun token n'a fonctionné, nettoyer
            if (!profileLoaded && !accessToken && !refreshToken) {
              await clearDriverStorage();
              if (isMounted) {
                setDriver(null);
                setDriverToken(null);
              }
            }
          } finally {
            // Toujours arrêter le chargement à la fin
            if (isMounted) {
              setDriverLoading(false);
            }
          }
        }

        const [enterpriseToken, enterpriseSessionRaw, justCreated] = await Promise.all([
          AsyncStorage.getItem(ENTERPRISE_TOKEN_KEY),
          AsyncStorage.getItem(ENTERPRISE_SESSION_KEY),
          AsyncStorage.getItem("enterprise_session_just_created"),
        ]);
        if (enterpriseToken && enterpriseSessionRaw) {
          try {
            const parsed: EnterpriseSessionState =
              JSON.parse(enterpriseSessionRaw);
            // Restaurer la session depuis le stockage
            setEnterpriseSession({ ...parsed, token: enterpriseToken });

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
                const refreshToken = await AsyncStorage.getItem(ENTERPRISE_REFRESH_KEY) ?? parsed.refreshToken ?? null;
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
                  const refreshToken = await AsyncStorage.getItem(ENTERPRISE_REFRESH_KEY) ?? parsed.refreshToken;
                  if (refreshToken) {
                    try {
                      const refreshResponse = await refreshEnterpriseToken(refreshToken);
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
            await clearEnterpriseStorage();
            setEnterpriseSession(null);
          }
        }
      } finally {
        if (isMounted) setInitialLoading(false);
      }
    })();
    return () => {
      isMounted = false;
    };
  }, [clearDriverStorage, clearEnterpriseStorage, storeMode, handleEnterpriseSuccess]);

  // ✅ REFRESH PROACTIF : Rafraîchir le token driver 5 minutes avant expiration
  // Évite les erreurs 401 et améliore l'expérience utilisateur (comme WhatsApp)
  useEffect(() => {
    if (!driverToken || mode !== "driver") return;

    const expiresAt = getTokenExpiration(driverToken);
    if (!expiresAt) {
      console.warn("[useAuth] Impossible de décoder l'expiration du token driver");
      return;
    }

    const now = Date.now();
    const timeUntilExpiry = expiresAt - now;
    const refreshBeforeExpiry = 5 * 60 * 1000; // 5 minutes

    // Si le token expire dans plus de 5 minutes, planifier le refresh
    if (timeUntilExpiry > refreshBeforeExpiry) {
      const timeoutId = setTimeout(async () => {
        console.log("[useAuth] 🔄 Refresh proactif du token driver (5min avant expiration)");
        try {
          const refreshToken = await secureStorage.getRefreshToken();
          if (refreshToken) {
            const refreshResponse = await refreshAccessToken(refreshToken);

            // Stocker le nouveau access_token
            await secureStorage.setAccessToken(refreshResponse.access_token);
            setDriverToken(refreshResponse.access_token);

            // Mettre à jour refresh_token si rotation activée
            if (refreshResponse.refresh_token) {
              await secureStorage.setRefreshToken(refreshResponse.refresh_token);
            }

            // Invalider le cache de l'intercepteur pour forcer l'utilisation du nouveau token
            invalidateInterceptorCache();

            console.log("[useAuth] ✅ Refresh proactif réussi");
          }
        } catch (error: any) {
          const status = error?.response?.status;
          const errorData = error?.response?.data;

          // ✅ Si 403 (compte désactivé), forcer déconnexion immédiate
          if (status === 403) {
            console.error(
              "[useAuth] 🚫 Compte désactivé (403) lors du refresh proactif. Déconnexion forcée.",
              errorData
            );
            // Nettoyer le stockage et réinitialiser l'état
            await secureStorage.clearAll();
            await asyncStorage.clearAuth();
            setDriverToken(null);
            setDriver(null);
            invalidateInterceptorCache();
            return;
          }

          console.warn("[useAuth] ⚠️ Refresh proactif échoué (fallback sur intercepteur 401):", error);
          // Ne pas déconnecter l'utilisateur pour les autres erreurs, l'intercepteur gérera le 401
        }
      }, timeUntilExpiry - refreshBeforeExpiry);

      console.log(`[useAuth] ⏰ Refresh proactif planifié dans ${Math.round((timeUntilExpiry - refreshBeforeExpiry) / 1000 / 60)} minutes`);

      return () => clearTimeout(timeoutId);
    } else if (timeUntilExpiry > 0 && timeUntilExpiry <= refreshBeforeExpiry) {
      // Token expire bientôt (< 5min), rafraîchir immédiatement
      console.log("[useAuth] ⚡ Token expire dans moins de 5min, refresh immédiat");
      (async () => {
        try {
          const refreshToken = await secureStorage.getRefreshToken();
          if (refreshToken) {
            const refreshResponse = await refreshAccessToken(refreshToken);
            await secureStorage.setAccessToken(refreshResponse.access_token);
            setDriverToken(refreshResponse.access_token);
            if (refreshResponse.refresh_token) {
              await secureStorage.setRefreshToken(refreshResponse.refresh_token);
            }
            invalidateInterceptorCache();
            console.log("[useAuth] ✅ Refresh immédiat réussi");
          }
        } catch (error: any) {
          const status = error?.response?.status;
          const errorData = error?.response?.data;

          // ✅ Si 403 (compte désactivé), forcer déconnexion immédiate
          if (status === 403) {
            console.error(
              "[useAuth] 🚫 Compte désactivé (403) lors du refresh immédiat. Déconnexion forcée.",
              errorData
            );
            // Nettoyer le stockage et réinitialiser l'état
            await secureStorage.clearAll();
            await asyncStorage.clearAuth();
            setDriverToken(null);
            setDriver(null);
            invalidateInterceptorCache();
            return;
          }

          console.warn("[useAuth] ⚠️ Refresh immédiat échoué:", error);
        }
      })();
    }
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
      const timeoutId = setTimeout(async () => {
        console.log("[useAuth] 🔄 Refresh proactif du token entreprise (5min avant expiration)");
        try {
          const refreshToken = enterpriseSession.refreshToken || await AsyncStorage.getItem(ENTERPRISE_REFRESH_KEY);
          if (refreshToken) {
            const refreshResponse = await refreshEnterpriseToken(refreshToken);
            await handleEnterpriseSuccess(refreshResponse);
            console.log("[useAuth] ✅ Refresh proactif entreprise réussi");
          }
        } catch (error) {
          console.warn("[useAuth] ⚠️ Refresh proactif entreprise échoué:", error);
        }
      }, timeUntilExpiry - refreshBeforeExpiry);

      console.log(`[useAuth] ⏰ Refresh proactif entreprise planifié dans ${Math.round((timeUntilExpiry - refreshBeforeExpiry) / 1000 / 60)} minutes`);

      return () => clearTimeout(timeoutId);
    } else if (timeUntilExpiry > 0 && timeUntilExpiry <= refreshBeforeExpiry) {
      // Token expire bientôt (< 5min), rafraîchir immédiatement
      console.log("[useAuth] ⚡ Token entreprise expire dans moins de 5min, refresh immédiat");
      (async () => {
        try {
          const refreshToken = enterpriseSession.refreshToken || await AsyncStorage.getItem(ENTERPRISE_REFRESH_KEY);
          if (refreshToken) {
            const refreshResponse = await refreshEnterpriseToken(refreshToken);
            await handleEnterpriseSuccess(refreshResponse);
            console.log("[useAuth] ✅ Refresh immédiat entreprise réussi");
          }
        } catch (error) {
          console.warn("[useAuth] ⚠️ Refresh immédiat entreprise échoué:", error);
        }
      })();
    }
  }, [enterpriseSession?.token, mode, handleEnterpriseSuccess]);

  // ✅ Recharger la session entreprise après un switchMode vers "enterprise"
  // Ce useEffect se déclenche quand le mode change vers "enterprise" et qu'il n'y a pas encore de session chargée
  useEffect(() => {
    // #region agent log
    fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ location: 'useAuth.tsx:useEffect:switchMode', message: 'useEffect switchMode entry', data: { mode, hasEnterpriseSession: !!enterpriseSession, initialLoading }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'G' }) }).catch(() => { });
    // #endregion

    if (mode !== "enterprise" || enterpriseSession || initialLoading) {
      // #region agent log
      if (mode === "enterprise") {
        fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ location: 'useAuth.tsx:useEffect:switchMode', message: 'useEffect switchMode skipped', data: { reason: enterpriseSession ? 'hasSession' : initialLoading ? 'loading' : 'notEnterprise' }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'G' }) }).catch(() => { });
      }
      // #endregion
      return;
    }

    let isMounted = true;
    (async () => {
      try {
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ location: 'useAuth.tsx:useEffect:switchMode', message: 'useEffect switchMode loading session', data: {}, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'G' }) }).catch(() => { });
        // #endregion

        const [enterpriseToken, enterpriseSessionRaw, justCreated] = await Promise.all([
          AsyncStorage.getItem(ENTERPRISE_TOKEN_KEY),
          AsyncStorage.getItem(ENTERPRISE_SESSION_KEY),
          AsyncStorage.getItem("enterprise_session_just_created"),
        ]);

        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ location: 'useAuth.tsx:useEffect:switchMode', message: 'useEffect switchMode session loaded', data: { hasToken: !!enterpriseToken, hasSession: !!enterpriseSessionRaw, justCreated }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'G' }) }).catch(() => { });
        // #endregion

        if (enterpriseToken && enterpriseSessionRaw && isMounted) {
          try {
            const parsed: EnterpriseSessionState = JSON.parse(enterpriseSessionRaw);
            // Restaurer la session depuis le stockage
            setEnterpriseSession({ ...parsed, token: enterpriseToken });

            // #region agent log
            fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ location: 'useAuth.tsx:useEffect:switchMode', message: 'useEffect switchMode session set', data: { hasToken: !!enterpriseToken, companyId: parsed.company?.id }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'G' }) }).catch(() => { });
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
            fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ location: 'useAuth.tsx:useEffect:switchMode', message: 'useEffect switchMode error', data: { error: String(error) }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'G' }) }).catch(() => { });
            // #endregion
          }
        }
      } catch (error) {
        // eslint-disable-next-line no-console
        console.warn("[ENT] Erreur lors de la lecture AsyncStorage après switchMode:", error);
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ location: 'useAuth.tsx:useEffect:switchMode', message: 'useEffect switchMode AsyncStorage error', data: { error: String(error) }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'G' }) }).catch(() => { });
        // #endregion
      }
    })();

    return () => {
      isMounted = false;
    };
  }, [mode, enterpriseSession, initialLoading]);

  const login = useCallback(
    async (email: string, password: string) => {
      setDriverLoading(true);
      try {
        const response = await loginDriver(email, password);
        await handleDriverLoginSuccess(response);
      } finally {
        setDriverLoading(false);
      }
    },
    [handleDriverLoginSuccess]
  );

  const logout = useCallback(async () => {
    setDriverLoading(true);
    try {
      // Appeler /auth/logout pour invalider server-side
      const accessToken = await secureStorage.getAccessToken();
      const refreshToken = await secureStorage.getRefreshToken();

      if (accessToken) {
        try {
          await api.post(
            "/auth/logout",
            { refresh_token: refreshToken ?? null }, // ✅ Envoyer refresh_token si disponible
            {
              headers: { Authorization: `Bearer ${accessToken}` },
            }
          );
        } catch (error) {
          // Ignorer les erreurs de logout (token peut être déjà expiré)
          console.warn("Erreur lors du logout server-side:", error);
        }
      }

      // Nettoyer tout le stockage
      await secureStorage.clearAll();
      await asyncStorage.clearAuth();

      // ⚡ OPTIMISATION : Invalider le cache de l'intercepteur lors du logout
      invalidateInterceptorCache();

      // Reset state
      setDriver(null);
      setDriverToken(null);
    } finally {
      setDriverLoading(false);
    }
  }, []);

  // ✅ Fonction pour charger la session driver depuis SecureStorage sans faire de requête API
  // Utile après un switchMode quand la session vient d'être créée
  const loadDriverSession = useCallback(async () => {
    // #region agent log
    fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ location: 'useAuth.tsx:loadDriverSession', message: 'loadDriverSession entry', data: {}, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'K' }) }).catch(() => { });
    // #endregion

    setDriverLoading(true);
    try {
      const [accessToken, refreshToken, userPublicId] = await Promise.all([
        secureStorage.getAccessToken(),
        secureStorage.getRefreshToken(),
        secureStorage.getUserPublicId(),
      ]);

      // #region agent log
      fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ location: 'useAuth.tsx:loadDriverSession', message: 'loadDriverSession loaded', data: { hasToken: !!accessToken, hasRefreshToken: !!refreshToken, hasUserPublicId: !!userPublicId }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'K' }) }).catch(() => { });
      // #endregion

      if (accessToken) {
        setDriverToken(accessToken);

        // Charger le profil driver pour mettre à jour le contexte
        try {
          const profile = await fetchDriverProfile();
          setDriver(profile);
          await asyncStorage.setDriverId(profile.id);

          // #region agent log
          fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ location: 'useAuth.tsx:loadDriverSession', message: 'loadDriverSession set', data: { hasToken: !!accessToken, driverId: profile.id }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'K' }) }).catch(() => { });
          // #endregion

          console.log("[useAuth] Driver session loaded from SecureStorage:", {
            hasToken: !!accessToken,
            hasRefreshToken: !!refreshToken,
            driverId: profile.id,
          });
        } catch (profileError) {
          console.warn("[useAuth] Error loading driver profile:", profileError);
          // #region agent log
          fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ location: 'useAuth.tsx:loadDriverSession', message: 'loadDriverSession profile error', data: { error: String(profileError) }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'K' }) }).catch(() => { });
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
      fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ location: 'useAuth.tsx:loadDriverSession', message: 'loadDriverSession error', data: { error: String(error) }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'K' }) }).catch(() => { });
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
    } catch (error) {
      console.warn("Erreur rafraîchissement profil chauffeur :", error);
      await logout();
    } finally {
      setDriverLoading(false);
    }
  }, [driverToken, logout]);

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
    fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ location: 'useAuth.tsx:loadEnterpriseSession', message: 'loadEnterpriseSession entry', data: {}, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'H' }) }).catch(() => { });
    // #endregion

    setEnterpriseLoading(true);
    try {
      const [enterpriseToken, enterpriseSessionRaw] = await Promise.all([
        AsyncStorage.getItem(ENTERPRISE_TOKEN_KEY),
        AsyncStorage.getItem(ENTERPRISE_SESSION_KEY),
      ]);

      // #region agent log
      fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ location: 'useAuth.tsx:loadEnterpriseSession', message: 'loadEnterpriseSession loaded', data: { hasToken: !!enterpriseToken, hasSession: !!enterpriseSessionRaw }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'H' }) }).catch(() => { });
      // #endregion

      if (enterpriseToken && enterpriseSessionRaw) {
        const parsed: EnterpriseSessionState = JSON.parse(enterpriseSessionRaw);
        setEnterpriseSession({ ...parsed, token: enterpriseToken });

        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ location: 'useAuth.tsx:loadEnterpriseSession', message: 'loadEnterpriseSession set', data: { hasToken: !!enterpriseToken, companyId: parsed.company?.id }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'H' }) }).catch(() => { });
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
      fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ location: 'useAuth.tsx:loadEnterpriseSession', message: 'loadEnterpriseSession error', data: { error: String(error) }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'H' }) }).catch(() => { });
      // #endregion
      setEnterpriseSession(null);
    } finally {
      setEnterpriseLoading(false);
    }
  }, []);

  const refreshEnterprise = useCallback(async () => {
    const refreshToken = await AsyncStorage.getItem(ENTERPRISE_REFRESH_KEY);
    if (!refreshToken) return;
    try {
      const response = await refreshEnterpriseToken(refreshToken);
      await handleEnterpriseSuccess(response);
    } catch (error) {
      console.warn("Refresh token entreprise invalide :", error);
      await clearEnterpriseStorage();
      setEnterpriseSession(null);
    }
  }, [clearEnterpriseStorage, handleEnterpriseSuccess]);

  const logoutEnterprise = useCallback(async () => {
    await clearEnterpriseStorage();
    setEnterpriseSession(null);
    setPendingEnterpriseMfa(null);
  }, [clearEnterpriseStorage]);

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
