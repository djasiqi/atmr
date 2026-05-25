import { isFeatureEnabled } from "../../core/featureFlags/registry";

function envTrue(value: string | undefined): boolean {
  if (!value) return false;
  const normalized = value.trim().toLowerCase();
  return normalized === "1" || normalized === "true" || normalized === "yes" || normalized === "on";
}

export function isPerfChatLocalPatchEnabled(): boolean {
  // Hard override for perf protocol sessions, even if runtime feature overrides disable it.
  if (envTrue(process.env.EXPO_PUBLIC_PERF_CHAT_LOCAL_PATCH_FORCE)) return true;
  if (envTrue(process.env.EXPO_PUBLIC_PERF_CHAT_LOCAL_PATCH)) return true;
  return isFeatureEnabled("perf_chat_local_patch_enabled");
}
