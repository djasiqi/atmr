import { getRuntimeNumericFlag, isFeatureEnabled } from "../../../core/featureFlags/registry";
import { MAX_BATCH_AGE_MS, REALTIME_FLUSH_MS } from "./gpsFlushConstants";

/** Priorité : override bootstrap > env > constante legacy. */
export function resolveRealtimeFlushMs(): number {
  const runtime = getRuntimeNumericFlag("company_map_realtime_flush_ms");
  if (runtime != null && runtime > 0) return runtime;
  const envRaw = process.env.EXPO_PUBLIC_COMPANY_MAP_REALTIME_FLUSH_MS;
  const envParsed = envRaw != null ? parseInt(envRaw, 10) : NaN;
  if (Number.isFinite(envParsed) && envParsed > 0) return envParsed;
  return REALTIME_FLUSH_MS;
}

export function resolveMaxBatchAgeMs(): number {
  const flush = resolveRealtimeFlushMs();
  if (flush <= 100) return 300;
  if (flush <= 250) return 500;
  return MAX_BATCH_AGE_MS;
}

export function isCompanyMapDynamicFilterEnabled(): boolean {
  if (process.env.EXPO_PUBLIC_COMPANY_MAP_DYNAMIC_FILTER_ENABLED === "0") return false;
  return isFeatureEnabled("company_map_dynamic_filter_enabled");
}

export function isCompanyMapAutofitStructuralOnly(): boolean {
  if (process.env.EXPO_PUBLIC_COMPANY_MAP_AUTOFIT_STRUCTURAL_ONLY === "0") return false;
  return isFeatureEnabled("company_map_autofit_structural_only");
}

export function isMobileMapParityMode(): boolean {
  return isFeatureEnabled("mobile_map_parity_mode");
}
