import type { DriverProfile } from "../api";

/** true/false si le profil porte une valeur ; null si inconnue. */
export function resolveDriverAvailabilityFromProfile(
  profile: DriverProfile | null | undefined
): boolean | null {
  if (!profile) return null;
  const raw = profile.is_available;
  if (typeof raw === "boolean") return raw;
  if (raw === 1 || raw === "1") return true;
  if (raw === 0 || raw === "0") return false;
  if (typeof raw === "string") {
    const normalized = raw.trim().toLowerCase();
    if (normalized === "true" || normalized === "yes") return true;
    if (normalized === "false" || normalized === "no") return false;
  }
  return null;
}

export function normalizeDriverProfilePayload(data: unknown): DriverProfile {
  if (!data || typeof data !== "object") return {};
  const raw = data as Record<string, unknown>;
  const nested = raw.profile;
  if (nested && typeof nested === "object") {
    return nested as DriverProfile;
  }
  return raw as DriverProfile;
}
