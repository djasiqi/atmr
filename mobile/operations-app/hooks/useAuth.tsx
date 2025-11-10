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
  AuthResponse,
  Driver,
  fetchDriverProfile,
  loginDriver,
} from "@/services/api";
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

const DRIVER_TOKEN_KEY = "token";
const DRIVER_ID_KEY = "driver_id";
const MODE_KEY = "auth.mode";
const ENTERPRISE_DEVICE_KEY = "enterprise.device_id";

type AuthMode = "driver" | "enterprise";

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
    await AsyncStorage.removeItem(DRIVER_TOKEN_KEY);
    await AsyncStorage.removeItem(DRIVER_ID_KEY);
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
      setDriverToken(response.token);
      await AsyncStorage.setItem(DRIVER_TOKEN_KEY, response.token);
      await storeMode("driver");
      try {
        const profile = await fetchDriverProfile();
        setDriver(profile);
        await AsyncStorage.setItem(DRIVER_ID_KEY, String(profile.id));
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
      await AsyncStorage.multiSet([
        [ENTERPRISE_TOKEN_KEY, session.token],
        [ENTERPRISE_SESSION_KEY, JSON.stringify(session)],
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
    },
    [storeMode]
  );

  useEffect(() => {
    let isMounted = true;
    (async () => {
      try {
        const storedMode = await AsyncStorage.getItem(MODE_KEY);
        if (storedMode === "driver" || storedMode === "enterprise") {
          setModeState(storedMode);
        } else {
          await AsyncStorage.setItem(MODE_KEY, "enterprise");
        }
        const storedDevice = await AsyncStorage.getItem(ENTERPRISE_DEVICE_KEY);
        if (storedDevice) setDeviceId(storedDevice);

        const storedDriverToken = await AsyncStorage.getItem(DRIVER_TOKEN_KEY);
        if (storedDriverToken && (storedMode === "driver" || !storedMode)) {
          setDriverToken(storedDriverToken);
          try {
            const profile = await fetchDriverProfile();
            if (isMounted) setDriver(profile);
          } catch (error) {
            console.warn("Token chauffeur invalide :", error);
            await clearDriverStorage();
            if (isMounted) {
              setDriver(null);
              setDriverToken(null);
            }
          }
        }

        const [enterpriseToken, enterpriseSessionRaw] = await Promise.all([
          AsyncStorage.getItem(ENTERPRISE_TOKEN_KEY),
          AsyncStorage.getItem(ENTERPRISE_SESSION_KEY),
        ]);
        if (enterpriseToken && enterpriseSessionRaw) {
          try {
            const parsed: EnterpriseSessionState =
              JSON.parse(enterpriseSessionRaw);
            setEnterpriseSession({ ...parsed, token: enterpriseToken });
            const latest = await fetchEnterpriseSession(enterpriseToken);
            const updated: EnterpriseSessionState = {
              token: enterpriseToken,
              refreshToken:
                (await AsyncStorage.getItem(ENTERPRISE_REFRESH_KEY)) ??
                parsed.refreshToken ??
                null,
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
          } catch (error) {
            console.warn("Session entreprise invalide :", error);
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
  }, [clearDriverStorage, clearEnterpriseStorage]);

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
    await clearDriverStorage();
    setDriver(null);
    setDriverToken(null);
  }, [clearDriverStorage]);

  const refreshProfile = useCallback(async () => {
    if (!driverToken) return;
    setDriverLoading(true);
    try {
      const profile = await fetchDriverProfile();
      setDriver(profile);
      await AsyncStorage.setItem(DRIVER_ID_KEY, String(profile.id));
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
        return { mfaRequired: false as const };
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
