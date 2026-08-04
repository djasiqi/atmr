import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from 'react';
import apiClient from '../utils/apiClient';
import { getAuthEnv, getEnvUser, setEnvUser } from '../utils/webAuthSession';
import {
  clearExplicitLogoutMarker,
  endLoginSession,
  hasRecentExplicitLogout,
  isExplicitLogoutInProgress,
  isLoginSessionInProgress,
  AUTH_LOGOUT_AT_KEY,
} from '../utils/sessionLogoutState';

const SessionBootstrapContext = createContext({
  status: 'idle',
  isAuthenticated: false,
  user: null,
  refreshBootstrap: () => {},
  hydrateFromLogin: () => {},
});

export function SessionBootstrapProvider({ children }) {
  const [status, setStatus] = useState('idle');
  const [user, setUser] = useState(null);

  /** Applique immédiatement la session écrite par le login (évite Unauthorized). */
  const hydrateFromLogin = useCallback((nextUser) => {
    if (!nextUser) return;
    setUser(nextUser);
    setStatus('authenticated');
  }, []);

  const refreshBootstrap = useCallback(async () => {
    if (isExplicitLogoutInProgress()) {
      setUser(null);
      setStatus('anonymous');
      return false;
    }

    const cachedUser = getEnvUser(getAuthEnv());
    const skipSessionRestore = hasRecentExplicitLogout();

    if (skipSessionRestore && !cachedUser) {
      setUser(null);
      setStatus('anonymous');
      return false;
    }

    if (cachedUser && !skipSessionRestore) {
      setUser(cachedUser);
      setStatus('authenticated');
    } else {
      setStatus('loading');
    }

    try {
      const response = await apiClient.get('/auth/me', {
        skipAuthRedirect: false,
        skipFreshTokenLogout: true,
      });
      const payload = response?.data?.user || response?.data;
      if (payload) {
        if (skipSessionRestore) {
          setUser(null);
          setStatus('anonymous');
          clearExplicitLogoutMarker();
          return false;
        }
        const nextUser = payload?.role
          ? payload
          : { ...(cachedUser || {}), ...payload, role: cachedUser?.role };
        setEnvUser(nextUser, getAuthEnv(), { mirrorLegacy: true });
        setUser(nextUser);
        setStatus('authenticated');
        return true;
      }
      setUser(cachedUser || null);
      setStatus(cachedUser ? 'authenticated' : 'anonymous');
      return Boolean(cachedUser);
    } catch (error) {
      if (error?.response?.status === 401) {
        if (isLoginSessionInProgress()) {
          return Boolean(cachedUser);
        }
        setEnvUser(null, getAuthEnv(), { mirrorLegacy: true });
        setUser(null);
        setStatus('anonymous');
        clearExplicitLogoutMarker();
        return false;
      }
      setUser(cachedUser || null);
      setStatus(cachedUser ? 'authenticated' : 'error');
      return Boolean(cachedUser);
    } finally {
      if (isLoginSessionInProgress()) {
        endLoginSession();
      }
    }
  }, []);

  useEffect(() => {
    void refreshBootstrap();

    const onAuthChanged = () => {
      void refreshBootstrap();
    };

    const onStorage = (event) => {
      if (event.key === AUTH_LOGOUT_AT_KEY) {
        setUser(null);
        setStatus('anonymous');
        void import('../utils/deferredSessionLogout')
          .then(({ stopSessionIdleGuard }) => {
            stopSessionIdleGuard();
          })
          .catch(() => {});
        return;
      }
      if (
        event.key === null ||
        event.key?.endsWith('_user') ||
        event.key?.endsWith('_access_token') ||
        event.key === 'user' ||
        event.key === 'authToken' ||
        event.key === 'lirie_auth_env'
      ) {
        void refreshBootstrap();
      }
    };

    window.addEventListener('auth-changed', onAuthChanged);
    window.addEventListener('storage', onStorage);
    return () => {
      window.removeEventListener('auth-changed', onAuthChanged);
      window.removeEventListener('storage', onStorage);
    };
  }, [refreshBootstrap]);

  const value = useMemo(
    () => ({
      status,
      isAuthenticated: status === 'authenticated',
      user,
      refreshBootstrap,
      hydrateFromLogin,
    }),
    [status, user, refreshBootstrap, hydrateFromLogin]
  );

  return (
    <SessionBootstrapContext.Provider value={value}>
      {children}
    </SessionBootstrapContext.Provider>
  );
}

export function useSessionBootstrap() {
  return useContext(SessionBootstrapContext);
}

export default SessionBootstrapContext;
