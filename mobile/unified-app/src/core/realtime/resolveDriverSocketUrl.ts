import { getResolvedApiBaseUrl } from "../api/client";

/** URL Socket.IO chauffeur : env dédiée, repli legacy EXPO_PUBLIC_SOCKET_URL, sinon origine API. */
export function resolveDriverSocketUrl(): string | null {
  const fromEnv =
    process.env.EXPO_PUBLIC_DRIVER_SOCKET_URL?.trim() ||
    process.env.EXPO_PUBLIC_SOCKET_URL?.trim();
  if (fromEnv) return fromEnv;
  try {
    return new URL(getResolvedApiBaseUrl()).origin;
  } catch {
    return null;
  }
}
