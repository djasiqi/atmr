import { api, getApiErrorMessage, setMemoryAccessToken } from '@/services/api';
import {
  clearStoredTokens,
  getStoredAccessToken,
  getStoredRefreshToken,
  setStoredTokens,
} from '@/services/tokenStorage';

export type AuthMeUser = {
  id: number;
  public_id: string;
  username: string;
  email: string | null;
  role: string;
  bootstrap_version?: number;
  account_active?: boolean;
  profile_active?: boolean | null;
  profile_type?: string | null;
  message?: string | null;
  access_denied_code?: string | null;
};

export type LoginResponse = {
  message?: string;
  token: string;
  refresh_token?: string;
  user: {
    id: number;
    public_id: string;
    username: string;
    email: string | null;
    role: string;
    force_password_change?: boolean;
  };
};

export function mapBackendRoleToAppRole(
  role: string | null | undefined,
): 'client' | 'institution' | null {
  const r = String(role ?? '').trim().toLowerCase();
  if (r === 'client') {
    return 'client';
  }
  if (r === 'institution') {
    return 'institution';
  }
  return null;
}

export async function fetchAuthMe(): Promise<AuthMeUser> {
  const res = await api.get<AuthMeUser>('/auth/me');
  return res.data;
}

export async function loginWithPassword(
  email: string,
  password: string,
): Promise<LoginResponse> {
  const res = await api.post<LoginResponse>('/auth/login', { email, password });
  return res.data;
}

export async function persistSessionFromLogin(data: LoginResponse): Promise<void> {
  await setStoredTokens(data.token, data.refresh_token ?? null);
  setMemoryAccessToken(data.token);
}

export async function refreshSessionWithStoredRefresh(): Promise<boolean> {
  const rt = await getStoredRefreshToken();
  if (!rt) {
    return false;
  }
  try {
    const res = await api.post<{
      access_token: string;
      refresh_token?: string;
    }>('/auth/refresh-token', { refresh_token: rt });
    const { access_token, refresh_token } = res.data;
    setMemoryAccessToken(access_token);
    await setStoredTokens(access_token, refresh_token ?? rt);
    return true;
  } catch {
    setMemoryAccessToken(null);
    await clearStoredTokens();
    return false;
  }
}

export async function logoutRemote(): Promise<void> {
  try {
    await api.post('/auth/logout', {});
  } catch {
    // déconnexion locale même si le serveur échoue
  }
}

export async function clearLocalSession(): Promise<void> {
  setMemoryAccessToken(null);
  await clearStoredTokens();
}

export type BootstrapResult =
  | { kind: 'none' }
  | { kind: 'ok'; me: AuthMeUser }
  | { kind: 'network'; message: string }
  | { kind: 'forbidden'; message: string }
  | { kind: 'invalid_session' };

export async function bootstrapFromStorage(): Promise<BootstrapResult> {
  const access = await getStoredAccessToken();
  if (!access) {
    return { kind: 'none' };
  }
  setMemoryAccessToken(access);

  const tryMe = async (): Promise<AuthMeUser> => fetchAuthMe();

  try {
    const me = await tryMe();
    return { kind: 'ok', me };
  } catch (first: unknown) {
    const status = (first as { response?: { status?: number } })?.response?.status;
    if (status === 401) {
      const refreshed = await refreshSessionWithStoredRefresh();
      if (!refreshed) {
        return { kind: 'invalid_session' };
      }
      try {
        const me = await tryMe();
        return { kind: 'ok', me };
      } catch {
        return { kind: 'invalid_session' };
      }
    }
    if (status === 403) {
      return {
        kind: 'forbidden',
        message: getApiErrorMessage(first),
      };
    }
    if (status === 404) {
      return { kind: 'invalid_session' };
    }
    return {
      kind: 'network',
      message: getApiErrorMessage(first),
    };
  }
}
