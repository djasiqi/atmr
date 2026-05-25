import { isFeatureEnabled } from "../featureFlags/registry";

export type PerfInstrumentationTier = "off" | "dev" | "staging" | "prod_sampled";

let sessionSampled: boolean | null = null;

function resolveTierFromEnv(): PerfInstrumentationTier {
  if (!isFeatureEnabled("perf_instrumentation_enabled")) return "off";
  const explicit = process.env.EXPO_PUBLIC_PERF_INSTRUMENTATION_TIER?.trim().toLowerCase();
  if (explicit === "off" || explicit === "0") return "off";
  if (explicit === "staging") return "staging";
  if (explicit === "prod_sampled" || explicit === "prod") return "prod_sampled";
  if (explicit === "dev") return "dev";
  if (typeof __DEV__ !== "undefined" && __DEV__) return "dev";
  if (process.env.EXPO_PUBLIC_PERF_INSTRUMENTATION === "1") return "staging";
  return "off";
}

function resolveSessionSampled(tier: PerfInstrumentationTier): boolean {
  if (tier === "off") return false;
  if (tier === "dev") return true;
  if (sessionSampled !== null) return sessionSampled;
  const rate =
    tier === "staging"
      ? Number(process.env.EXPO_PUBLIC_PERF_SAMPLE_RATE_STAGING ?? "0.5")
      : Number(process.env.EXPO_PUBLIC_PERF_SAMPLE_RATE_PROD ?? "0.075");
  const clamped = Math.min(1, Math.max(0, rate));
  sessionSampled = Math.random() < clamped;
  return sessionSampled;
}

let cachedTier: PerfInstrumentationTier | null = null;

export function getPerfInstrumentationTier(): PerfInstrumentationTier {
  if (cachedTier === null) {
    cachedTier = resolveTierFromEnv();
  }
  return cachedTier;
}

export function isPerfInstrumentationActive(): boolean {
  const tier = getPerfInstrumentationTier();
  if (tier === "off") return false;
  return resolveSessionSampled(tier);
}

/** DEV: every event; staging/prod: aggregate windows only (no per-call emit). */
export function shouldEmitPerfEventPerCall(): boolean {
  return getPerfInstrumentationTier() === "dev" && isPerfInstrumentationActive();
}

export function shouldRecordPerfMetric(): boolean {
  return isPerfInstrumentationActive();
}

export function resetPerfInstrumentationTierForTests(): void {
  cachedTier = null;
  sessionSampled = null;
}

export function getPerfSessionSampledForTests(): boolean | null {
  return sessionSampled;
}
