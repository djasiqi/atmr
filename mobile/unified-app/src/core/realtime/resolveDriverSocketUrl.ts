import { getResolvedApiBaseUrl } from "../api/client";

/** URL Socket.IO chauffeur : env dédiée, sinon origine de l’API (même hôte que Flask). */
export function resolveDriverSocketUrl(): string | null {
  const fromEnv = process.env.EXPO_PUBLIC_DRIVER_SOCKET_URL;
  if (fromEnv?.trim()) return fromEnv.trim();
  try {
    return new URL(getResolvedApiBaseUrl()).origin;
  } catch {
    return null;
  }
}
