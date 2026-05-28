import { getResolvedApiBaseUrl } from "../api/client";

const DEFAULT_PROD_DRIVER_SOCKET_URL = "https://api.lirie.ch";

function isDevOrPrivateHost(hostname: string): boolean {
  const h = hostname.trim().toLowerCase();
  if (h === "localhost" || h === "127.0.0.1" || h === "[::1]" || h === "::1") return true;
  if (h === "10.0.2.2") return true;
  if (/^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$/.test(h)) return true;
  if (/^192\.168\.\d{1,3}\.\d{1,3}$/.test(h)) return true;
  const m = /^172\.(\d{1,3})\.\d{1,3}\.\d{1,3}$/.exec(h);
  if (!m) return false;
  const second = Number(m[1]);
  return second >= 16 && second <= 31;
}

function isUnsafeProductionSocketUrl(value: string): boolean {
  if (!value.startsWith("https://") && !value.startsWith("wss://")) return true;
  try {
    return isDevOrPrivateHost(new URL(value).hostname);
  } catch {
    return true;
  }
}

/** URL Socket.IO chauffeur : env dédiée, repli legacy EXPO_PUBLIC_SOCKET_URL, sinon origine API. */
export function resolveDriverSocketUrl(): string | null {
  const fromEnv =
    process.env.EXPO_PUBLIC_DRIVER_SOCKET_URL?.trim() ||
    process.env.EXPO_PUBLIC_SOCKET_URL?.trim();
  if (fromEnv) {
    return !__DEV__ && isUnsafeProductionSocketUrl(fromEnv)
      ? DEFAULT_PROD_DRIVER_SOCKET_URL
      : fromEnv;
  }
  try {
    return new URL(getResolvedApiBaseUrl()).origin;
  } catch {
    return null;
  }
}
