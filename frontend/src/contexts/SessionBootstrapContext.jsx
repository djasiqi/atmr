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
} from '../utils/sessionLogoutState';

const SessionBootstrapContext = createContext({
  status: 'idle',
  isAuthenticated: false,
  user: null,
  refreshBootstrap: () => {},
});

export function SessionBootstrapProvider({ children }) {
  const [status, setStatus] = useState('idle');
  const [user, setUser] = useState(null);

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
    window.addEventListener('auth-changed', onAuthChanged);
    return () => window.removeEventListener('auth-changed', onAuthChanged);
  }, [refreshBootstrap]);

  const value = useMemo(
    () => ({
      status,
      isAuthenticated: status === 'authenticated',
      user,
      refreshBootstrap,
    }),
    [status, user, refreshBootstrap]
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
