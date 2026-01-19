import React, {
  createContext,
  ReactNode,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import { AppState, AppStateStatus } from "react-native";
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
// ✅ CORRECTION : Les tokens Enterprise sont maintenant dans SecureStore
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
  invalidateEnterpriseInterceptorCache,
} from "@/services/enterpriseAuth";
import { notifyAuthReady, notifyAuthNotReady } from "@/services/authSync";
// ✅ PHASE 2 : Import de l'authentification biométrique
import { autoLoginWithBiometric } from "@/services/biometricAuth";

// ✅ Helper pour les logs de debug (dev uniquement)
// Évite les warnings de connexion en production
const debugLog = (data: any) => {
  if (__DEV__) {
    try {
      fetch("http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      }).catch(() => {
        // Ignorer silencieusement les erreurs de connexion au service de debug
      });
    } catch (e) {
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
    // ✅ CORRECTION : Utiliser SecureStore pour les tokens
    await secureStorage.clearEnterpriseTokens();
    // Garder AsyncStorage uniquement pour la session complète (données non sensibles)
    await AsyncStorage.removeItem(ENTERPRISE_SESSION_KEY);
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
    (async () => {
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
            if (!profileLoaded && driverRefreshToken) {
              try {
                const refreshResponse = await refreshAccessToken(driverRefreshToken);

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

                // ✅ PHASE 2 : Si le refresh token échoue, essayer auto-login avec authentification biométrique
                if (!profileLoaded) {
                  try {
                    // Tenter l'auto-login avec authentification biométrique
                    const biometricSuccess = await autoLoginWithBiometric({
                      promptMessage: "Authentifiez-vous pour vous reconnecter",
                      cancelLabel: "Annuler",
                      disableDeviceFallback: false, // Permet code PIN si biométrie échoue
                      fallbackLabel: "Utiliser le code PIN",
                    });

                    if (biometricSuccess) {
                      // Si l'auto-login biométrique réussit, charger le profil
                      const profile = await fetchDriverProfile();
                      if (isMounted) {
                        setDriver(profile);
                        await asyncStorage.setDriverId(profile.id);
                        await storeMode("driver");
                        profileLoaded = true;
                        console.log("[useAuth] ✅ Auto-login réussi avec authentification biométrique");
                      }
                    } else {
                      // Si l'authentification biométrique échoue ou est annulée, ne pas nettoyer
                      // L'utilisateur pourra se reconnecter manuellement
                      if (__DEV__) {
                        console.log("[useAuth] ⚠️ Auto-login biométrique annulé ou échoué");
                      }
                    }
                  } catch (autoLoginError) {
                    console.warn("[useAuth] ⚠️ Auto-login avec authentification biométrique échoué:", autoLoginError);
                    // Si l'auto-login échoue aussi, nettoyer
                    await secureStorage.clearAll();
                    await asyncStorage.clearAuth();
                    if (isMounted) {
                      setDriver(null);
                      setDriverToken(null);
                    }
                  }

                  // Si toujours pas de profil chargé, nettoyer
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
            }

            // 3. Si aucun token n'a fonctionné, essayer auto-login avec authentification biométrique
            if (!profileLoaded && !driverAccessToken && !driverRefreshToken) {
              try {
                // Tenter l'auto-login avec authentification biométrique
                const biometricSuccess = await autoLoginWithBiometric({
                  promptMessage: "Authentifiez-vous pour vous reconnecter",
                  cancelLabel: "Annuler",
                  disableDeviceFallback: false, // Permet code PIN si biométrie échoue
                  fallbackLabel: "Utiliser le code PIN",
                });

                if (biometricSuccess) {
                  // Si l'auto-login biométrique réussit, charger le profil
                  const profile = await fetchDriverProfile();
                  if (isMounted) {
                    setDriver(profile);
                    await asyncStorage.setDriverId(profile.id);
                    await storeMode("driver");
                    profileLoaded = true;
                    console.log("[useAuth] ✅ Auto-login réussi avec authentification biométrique");
                  }
                } else {
                  // Si l'authentification biométrique échoue ou est annulée, nettoyer
                  await clearDriverStorage();
                  if (isMounted) {
                    setDriver(null);
                    setDriverToken(null);
                  }
                }
              } catch (autoLoginError) {
                console.warn("[useAuth] ⚠️ Auto-login avec authentification biométrique échoué:", autoLoginError);
                // Si l'auto-login échoue, nettoyer
                await clearDriverStorage();
                if (isMounted) {
                  setDriver(null);
                  setDriverToken(null);
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
            const parsed: EnterpriseSessionState =
              JSON.parse(enterpriseSessionRaw);
            // Restaurer la session depuis le stockage
            // ✅ CORRECTION : Utiliser enterpriseToken déjà lu (depuis SecureStore)
            setEnterpriseSession({
              ...parsed,
              token: enterpriseToken,
              refreshToken: enterpriseRefreshToken ?? parsed.refreshToken ?? null,
            });

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
        if (isMounted) {
          setInitialLoading(false);
          // ✅ CORRECTION #1 : Notifier que l'auth est prête
          // Permet aux intercepteurs de savoir que les tokens sont chargés
          notifyAuthReady();
        }
      }
    })();
    return () => {
      isMounted = false;
    };
  }, [clearDriverStorage, clearEnterpriseStorage, storeMode, handleEnterpriseSuccess]);

  // ✅ PHASE 3 : REFRESH PROACTIF amélioré : Rafraîchir le token driver 10 minutes avant expiration
  // Évite les erreurs 401 et améliore l'expérience utilisateur (comme WhatsApp)
  // Timing augmenté de 5 à 10 minutes pour plus de marge
  useEffect(() => {
    if (!driverToken || mode !== "driver") return;

    const expiresAt = getTokenExpiration(driverToken);
    if (!expiresAt) {
      console.warn("[useAuth] Impossible de décoder l'expiration du token driver");
      return;
    }

    const now = Date.now();
    const timeUntilExpiry = expiresAt - now;
    const refreshBeforeExpiry = 10 * 60 * 1000; // ✅ PHASE 3 : 10 minutes (au lieu de 5)

    // ✅ PHASE 3 : Si le token expire dans plus de 10 minutes, planifier le refresh
    if (timeUntilExpiry > refreshBeforeExpiry) {
      const timeoutId = setTimeout(async () => {
        console.log("[useAuth] 🔄 Refresh proactif du token driver (10min avant expiration)");

        // ✅ PHASE 3 : Système de retry avec backoff exponentiel
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

            // Stocker le nouveau access_token
            await secureStorage.setAccessToken(refreshResponse.access_token);
            setDriverToken(refreshResponse.access_token);

            // Mettre à jour refresh_token si rotation activée
            if (refreshResponse.refresh_token) {
              await secureStorage.setRefreshToken(refreshResponse.refresh_token);
            }

            // Invalider le cache de l'intercepteur pour forcer l'utilisation du nouveau token
            invalidateInterceptorCache();

            console.log(`[useAuth] ✅ Refresh proactif réussi${retryCount > 0 ? ` (après ${retryCount} tentative(s))` : ""}`);
            return; // Succès, sortir de la boucle
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

        // ✅ PHASE 3 : Si toutes les tentatives ont échoué, vérifier si c'est critique
        if (lastError) {
          const status = lastError?.response?.status;
          const errorData = lastError?.response?.data;

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

          console.warn("[useAuth] ⚠️ Refresh proactif échoué (fallback sur intercepteur 401):", lastError);
          // Ne pas déconnecter l'utilisateur pour les autres erreurs, l'intercepteur gérera le 401
        }
      }, timeUntilExpiry - refreshBeforeExpiry);

      console.log(`[useAuth] ⏰ Refresh proactif planifié dans ${Math.round((timeUntilExpiry - refreshBeforeExpiry) / 1000 / 60)} minutes`);

      return () => clearTimeout(timeoutId);
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
            console.log(`[useAuth] ✅ Refresh immédiat réussi${retryCount > 0 ? ` (après ${retryCount} tentative(s))` : ""}`);
            return; // Succès
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

        // Gérer les erreurs finales
        if (lastError) {
          const status = lastError?.response?.status;
          const errorData = lastError?.response?.data;

          if (status === 403) {
            console.error("[useAuth] 🚫 Compte désactivé (403) lors du refresh immédiat. Déconnexion forcée.", errorData);
            await secureStorage.clearAll();
            await asyncStorage.clearAuth();
            setDriverToken(null);
            setDriver(null);
            invalidateInterceptorCache();
            return;
          }

          console.warn("[useAuth] ⚠️ Refresh immédiat échoué:", lastError);
          // Ne pas déconnecter l'utilisateur pour les autres erreurs, l'intercepteur gérera le 401
        }
      })();
    }
  }, [driverToken, mode]);

  // ✅ PHASE 3 : Refresh automatique au retour au premier plan
  useEffect(() => {
    if (!driverToken || mode !== "driver") return;

    const handleAppStateChange = async (nextAppState: AppStateStatus) => {
      if (nextAppState === "active") {
        // Vérifier si le token est proche de l'expiration
        const expiresAt = getTokenExpiration(driverToken);
        if (!expiresAt) return;

        const now = Date.now();
        const timeUntilExpiry = expiresAt - now;
        const refreshThreshold = 15 * 60 * 1000; // 15 minutes

        // Si le token expire dans moins de 15 minutes, rafraîchir
        if (timeUntilExpiry > 0 && timeUntilExpiry < refreshThreshold) {
          console.log("[useAuth] 🔄 App revenue au premier plan, refresh du token si nécessaire...");
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
              console.log("[useAuth] ✅ Refresh au retour au premier plan réussi");
            }
          } catch (error: any) {
            const status = error?.response?.status;
            // Ne pas logger les erreurs réseau temporaires comme des erreurs critiques
            if (status === 401 || status === 403) {
              console.error("[useAuth] ❌ Refresh au retour au premier plan échoué (critique):", status);
            } else {
              console.warn("[useAuth] ⚠️ Refresh au retour au premier plan échoué (non critique):", error?.message);
            }
            // Ne pas déconnecter, l'intercepteur gérera si nécessaire
          }
        }
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
      const timeoutId = setTimeout(async () => {
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

          const refreshResponse = await refreshEnterpriseToken(refreshToken);
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

          // ✅ Si 403 (compte désactivé), forcer déconnexion immédiate
          if (status === 403) {
            console.error(
              "[useAuth] 🚫 Compte désactivé (403) lors du refresh proactif entreprise. Déconnexion forcée.",
              errorData
            );
            // Nettoyer le stockage et réinitialiser l'état
            await clearEnterpriseStorage();
            setEnterpriseSession(null);
            invalidateEnterpriseInterceptorCache();
            return;
          }

          // Autres erreurs (réseau, serveur, etc.) → ne pas déconnecter
          console.warn(
            `[useAuth] ⚠️ Refresh proactif entreprise échoué (status: ${status || "network"}): ${errorMessage}`
          );
          // Ne pas déconnecter l'utilisateur pour les autres erreurs, l'intercepteur gérera le 401
        }
      }, timeUntilExpiry - refreshBeforeExpiry);

      console.log(`[useAuth] ⏰ Refresh proactif entreprise planifié dans ${Math.round((timeUntilExpiry - refreshBeforeExpiry) / 1000 / 60)} minutes`);

      return () => clearTimeout(timeoutId);
    } else if (timeUntilExpiry > 0 && timeUntilExpiry <= refreshBeforeExpiry) {
      // Token expire bientôt (< 5min), rafraîchir immédiatement
      console.log("[useAuth] ⚡ Token entreprise expire dans moins de 5min, refresh immédiat");
      (async () => {
        try {
          // ✅ CORRECTION : Utiliser SecureStore pour le refresh token
          const refreshToken = enterpriseSession.refreshToken || await secureStorage.getEnterpriseRefreshToken();
          if (refreshToken) {
            const refreshResponse = await refreshEnterpriseToken(refreshToken);
            await handleEnterpriseSuccess(refreshResponse);

            // ⚡ CORRECTION : Invalider le cache interceptor pour forcer l'utilisation du nouveau token
            invalidateEnterpriseInterceptorCache();

            console.log("[useAuth] ✅ Refresh immédiat entreprise réussi");
          }
        } catch (error: any) {
          const status = error?.response?.status;
          const errorData = error?.response?.data;

          // ✅ Si 403 (compte désactivé), forcer déconnexion immédiate
          if (status === 403) {
            console.error(
              "[useAuth] 🚫 Compte désactivé (403) lors du refresh immédiat entreprise. Déconnexion forcée.",
              errorData
            );
            // Nettoyer le stockage et réinitialiser l'état
            await clearEnterpriseStorage();
            setEnterpriseSession(null);
            invalidateEnterpriseInterceptorCache();
            return;
          }

          console.warn("[useAuth] ⚠️ Refresh immédiat entreprise échoué:", error);
        }
      })();
    }
  }, [enterpriseSession?.token, mode, handleEnterpriseSuccess]);

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

        // ✅ PHASE 1 : Sauvegarder les identifiants si l'utilisateur a coché "Se souvenir de moi"
        if (rememberMe) {
          try {
            await secureStorage.saveCredentials(email, password);
            if (__DEV__) {
              console.log("[useAuth] ✅ Identifiants sauvegardés pour auto-login");
            }
          } catch (error) {
            console.warn("[useAuth] ⚠️ Erreur lors de la sauvegarde des identifiants:", error);
            // Ne pas bloquer le login si la sauvegarde échoue
          }
        } else {
          // Si l'utilisateur ne veut pas se souvenir, supprimer les identifiants existants
          await secureStorage.clearSavedCredentials();
        }
      } finally {
        setDriverLoading(false);
      }
    },
    [handleDriverLoginSuccess]
  );

  const logout = useCallback(async () => {
    setDriverLoading(true);
    // ✅ CORRECTION #1 : Notifier que l'auth n'est plus prête
    notifyAuthNotReady();
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

      // ✅ PHASE 1 : Supprimer les identifiants sauvegardés lors du logout
      await secureStorage.clearSavedCredentials();

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
    } catch (error: any) {
      const status = error?.response?.status;
      const isNetworkError = !error?.response; // Pas de réponse = erreur réseau

      // ✅ Ne déconnecter que si c'est vraiment un problème d'authentification
      // (401 = token invalide, 403 = compte désactivé)
      // Ne pas déconnecter pour erreurs réseau temporaires (timeout, pas de connexion)
      if (status === 401 || status === 403) {
        console.error(
          "[useAuth] ❌ Erreur authentification lors du refresh profil (status: %s). Déconnexion.",
          status
        );
        await logout();
      } else if (isNetworkError) {
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
      // ✅ CORRECTION : Utiliser SecureStore pour le token
      const [enterpriseToken, enterpriseSessionRaw] = await Promise.all([
        secureStorage.getEnterpriseToken(),
        AsyncStorage.getItem(ENTERPRISE_SESSION_KEY),
      ]);

      // #region agent log
      fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ location: 'useAuth.tsx:loadEnterpriseSession', message: 'loadEnterpriseSession loaded', data: { hasToken: !!enterpriseToken, hasSession: !!enterpriseSessionRaw }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'H' }) }).catch(() => { });
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
      fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ location: 'useAuth.tsx:loadEnterpriseSession', message: 'loadEnterpriseSession error', data: { error: String(error) }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'H' }) }).catch(() => { });
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
      const response = await refreshEnterpriseToken(refreshToken);
      await handleEnterpriseSuccess(response);
    } catch (error) {
      console.warn("Refresh token entreprise invalide :", error);
      await clearEnterpriseStorage();
      setEnterpriseSession(null);
    }
  }, [clearEnterpriseStorage, handleEnterpriseSuccess]);

  const logoutEnterprise = useCallback(async () => {
    // ✅ CORRECTION #1 : Notifier que l'auth n'est plus prête
    notifyAuthNotReady();
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
