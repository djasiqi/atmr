import { useRouter } from 'expo-router';
import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';

import { getApiErrorMessage } from '@/services/api';
import {
  bootstrapFromStorage,
  clearLocalSession,
  fetchAuthMe,
  loginWithPassword,
  logoutRemote,
  mapBackendRoleToAppRole,
  persistSessionFromLogin,
  type AuthMeUser,
} from '@/services/auth';

export type AuthStatus =
  | 'bootstrapping'
  | 'authenticated'
  | 'unauthenticated'
  | 'bootstrap_failed';

type AuthContextValue = {
  status: AuthStatus;
  role: 'client' | 'institution' | null;
  user: AuthMeUser | null;
  bootstrapError: string | null;
  login: (email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  refreshSession: () => Promise<void>;
  retryBootstrap: () => Promise<void>;
  goToLoginAfterBootstrapFailure: () => void;
};

const AuthContext = createContext<AuthContextValue | null>(null);

const BOOTSTRAP_TIMEOUT_MS = 12_000;

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const [status, setStatus] = useState<AuthStatus>('bootstrapping');
  const [user, setUser] = useState<AuthMeUser | null>(null);
  const [bootstrapError, setBootstrapError] = useState<string | null>(null);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const clearBootstrapTimer = useCallback(() => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
  }, []);

  const runBootstrap = useCallback(async () => {
    clearBootstrapTimer();
    setBootstrapError(null);
    setStatus('bootstrapping');

    timeoutRef.current = setTimeout(() => {
      setStatus((s) => {
        if (s === 'bootstrapping') {
          setBootstrapError('Connexion trop lente ou impossible. Vérifiez le réseau.');
          return 'bootstrap_failed';
        }
        return s;
      });
    }, BOOTSTRAP_TIMEOUT_MS);

    try {
      const result = await bootstrapFromStorage();
      clearBootstrapTimer();

      if (result.kind === 'none' || result.kind === 'invalid_session') {
        if (result.kind === 'invalid_session') {
          await clearLocalSession();
        }
        setUser(null);
        setStatus('unauthenticated');
        return;
      }

      if (result.kind === 'forbidden') {
        setUser(null);
        setBootstrapError(result.message);
        setStatus('bootstrap_failed');
        return;
      }

      if (result.kind === 'network') {
        setUser(null);
        setBootstrapError(result.message);
        setStatus('bootstrap_failed');
        return;
      }

      setUser(result.me);
      setStatus('authenticated');
    } catch (e) {
      clearBootstrapTimer();
      setUser(null);
      setBootstrapError(getApiErrorMessage(e));
      setStatus('bootstrap_failed');
    }
  }, [clearBootstrapTimer]);

  useEffect(() => {
    void runBootstrap();
    return () => {
      clearBootstrapTimer();
    };
  }, [runBootstrap, clearBootstrapTimer]);

  const role = useMemo(() => mapBackendRoleToAppRole(user?.role), [user?.role]);

  const login = useCallback(async (email: string, password: string) => {
    const data = await loginWithPassword(email.trim(), password);
    if (data.user?.force_password_change) {
      throw new Error('Changement de mot de passe requis (non géré sur mobile pour le moment).');
    }
    await persistSessionFromLogin(data);
    const me = await fetchAuthMe();
    setUser(me);
    setStatus('authenticated');
    const r = mapBackendRoleToAppRole(me.role);
    if (r === 'client') {
      router.replace('/(client)');
    } else if (r === 'institution') {
      router.replace('/(institution)');
    } else {
      router.replace('/');
    }
  }, [router]);

  const logout = useCallback(async () => {
    await logoutRemote();
    await clearLocalSession();
    setUser(null);
    setStatus('unauthenticated');
    router.replace('/(auth)/login');
  }, [router]);

  const refreshSession = useCallback(async () => {
    const me = await fetchAuthMe();
    setUser(me);
  }, []);

  const retryBootstrap = useCallback(async () => {
    await runBootstrap();
  }, [runBootstrap]);

  const goToLoginAfterBootstrapFailure = useCallback(() => {
    clearBootstrapTimer();
    void clearLocalSession();
    setUser(null);
    setBootstrapError(null);
    setStatus('unauthenticated');
    router.replace('/(auth)/login');
  }, [router, clearBootstrapTimer]);

  const value = useMemo<AuthContextValue>(
    () => ({
      status,
      role,
      user,
      bootstrapError,
      login,
      logout,
      refreshSession,
      retryBootstrap,
      goToLoginAfterBootstrapFailure,
    }),
    [
      status,
      role,
      user,
      bootstrapError,
      login,
      logout,
      refreshSession,
      retryBootstrap,
      goToLoginAfterBootstrapFailure,
    ],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error('useAuth doit être utilisé sous AuthProvider');
  }
  return ctx;
}
