type FeatureFlagSource = "external" | "env" | "static";

type FeatureFlagDefinition = {
  key: string;
  source: FeatureFlagSource;
  enabled: boolean;
  description: string;
};

function envEnabled(name: string): boolean {
  return process.env[name] === "1";
}

function envExplicitlyDisabled(name: string): boolean {
  return process.env[name] === "0";
}

// Single source of truth for rollout-safe toggles.
export const featureFlags = {
  driver_unified_enabled: {
    key: "driver_unified_enabled",
    source: "external",
    /** Opt-out: le backend peut forcer `false` dans le bootstrap (cohortes). Absence de clé = accès chauffeur unifié. */
    enabled: true,
    description: "Espace chauffeur dans l’app unifiée. Désactivable via feature_flags (bootstrap) pour cohorte legacy.",
  } satisfies FeatureFlagDefinition,
  ws_service_canary: {
    key: "ws_service_canary",
    source: "env",
    enabled: envEnabled("EXPO_PUBLIC_WS_CANARY_ENABLED"),
    description:
      "Route Socket.IO vers ws-service via header X-WS-Canary (rollout progressif PR C).",
  } satisfies FeatureFlagDefinition,
  realtime_socket_enabled: {
    key: "realtime_socket_enabled",
    source: "env",
    enabled: envExplicitlyDisabled("EXPO_PUBLIC_ENABLE_DRIVER_SOCKET")
      ? false
      : envEnabled("EXPO_PUBLIC_ENABLE_DRIVER_SOCKET") ||
        (typeof __DEV__ !== "undefined" && __DEV__),
    description:
      "Socket chauffeur (messagerie, missions). Local : actif par défaut en __DEV__ ou EXPO_PUBLIC_ENABLE_DRIVER_SOCKET=1. Voir expo-driver-dev.env.template.",
  } satisfies FeatureFlagDefinition,
  tracking_background_enabled: {
    key: "tracking_background_enabled",
    source: "env",
    enabled: envEnabled("EXPO_PUBLIC_ENABLE_BG_LOCATION"),
    description: "Allow background location capabilities/tracking flow.",
  } satisfies FeatureFlagDefinition,
  driver_mission_live_tracking_guard_enabled: {
    key: "driver_mission_live_tracking_guard_enabled",
    source: "env",
    enabled:
      process.env.EXPO_PUBLIC_ENABLE_DRIVER_MISSION_LIVE_TRACKING_GUARD === undefined
        ? true
        : envEnabled("EXPO_PUBLIC_ENABLE_DRIVER_MISSION_LIVE_TRACKING_GUARD"),
    description:
      "Gate store-safe EN_ROUTE/IN_PROGRESS sur capability tracking mission (disclosure + permissions).",
  } satisfies FeatureFlagDefinition,
  tracking_persistent_runtime_enabled: {
    key: "tracking_persistent_runtime_enabled",
    source: "env",
    enabled: envEnabled("EXPO_PUBLIC_ENABLE_TRACKING_PERSISTENT_QUEUE"),
    description: "Enable durable tracking queue + replay/flush delivery pipeline.",
  } satisfies FeatureFlagDefinition,
  tracking_presence_mode_enabled: {
    key: "tracking_presence_mode_enabled",
    source: "env",
    enabled: envEnabled("EXPO_PUBLIC_ENABLE_TRACKING_PRESENCE_MODE"),
    description: "Enable availability_presence tracking mode outside strict mission_live cadence.",
  } satisfies FeatureFlagDefinition,
  driver_tracking_work_window_enabled: {
    key: "driver_tracking_work_window_enabled",
    source: "env",
    /** Activé par défaut : règle métier ops (07h–19h presence, hors plage = mission only). */
    enabled:
      process.env.EXPO_PUBLIC_ENABLE_DRIVER_TRACKING_WORK_WINDOW === undefined
        ? true
        : envEnabled("EXPO_PUBLIC_ENABLE_DRIVER_TRACKING_WORK_WINDOW"),
    description:
      "Force le tracking GPS chauffeur en mode présence pendant la fenêtre 07h–19h (heure locale) même sans mission active. Hors plage : tracking uniquement si une mission éligible est en cours.",
  } satisfies FeatureFlagDefinition,
  tracking_http_fallback_enabled: {
    key: "tracking_http_fallback_enabled",
    source: "env",
    enabled: envEnabled("EXPO_PUBLIC_ENABLE_TRACKING_HTTP_FALLBACK"),
    description: "Allow HTTP fallback path when socket/ack transport is unavailable.",
  } satisfies FeatureFlagDefinition,
  tracking_resume_resync_enabled: {
    key: "tracking_resume_resync_enabled",
    source: "env",
    enabled: envEnabled("EXPO_PUBLIC_ENABLE_TRACKING_RESUME_RESYNC"),
    description: "Trigger queue flush + mission resync when app returns foreground.",
  } satisfies FeatureFlagDefinition,
  tracking_real_ack_semantics_enabled: {
    key: "tracking_real_ack_semantics_enabled",
    source: "env",
    enabled: envEnabled("EXPO_PUBLIC_ENABLE_TRACKING_REAL_ACK_SEMANTICS"),
    description: "Track delivery success only after backend ACK and not socket emit acceptance.",
  } satisfies FeatureFlagDefinition,
  tracking_queue_compaction_enabled: {
    key: "tracking_queue_compaction_enabled",
    source: "env",
    enabled: envEnabled("EXPO_PUBLIC_ENABLE_TRACKING_QUEUE_COMPACTION"),
    description: "Enable queue compaction strategy for prolonged offline sessions.",
  } satisfies FeatureFlagDefinition,
  tracking_safe_stale_fallback_enabled: {
    key: "tracking_safe_stale_fallback_enabled",
    source: "env",
    enabled: envEnabled("EXPO_PUBLIC_ENABLE_TRACKING_SAFE_STALE_FALLBACK"),
    description: "Enable timeout/circuit-breaker protections for stale location fallback.",
  } satisfies FeatureFlagDefinition,
  tracking_adaptive_cadence_enabled: {
    key: "tracking_adaptive_cadence_enabled",
    source: "env",
    enabled: envEnabled("EXPO_PUBLIC_ENABLE_TRACKING_ADAPTIVE_CADENCE"),
    description: "Enable network-aware cadence resolver for tracking loops and ACK stale policy.",
  } satisfies FeatureFlagDefinition,
  driver_runtime_decommission_enabled: {
    key: "driver_runtime_decommission_enabled",
    source: "env",
    enabled: envEnabled("EXPO_PUBLIC_ENABLE_DRIVER_RUNTIME_DECOMMISSION"),
    description: "Master switch for driver runtime replacement stack.",
  } satisfies FeatureFlagDefinition,
  driver_fcm_native_enabled: {
    key: "driver_fcm_native_enabled",
    source: "env",
    enabled: envEnabled("EXPO_PUBLIC_ENABLE_DRIVER_FCM_NATIVE"),
    description: "Enable native FCM driver integration path.",
  } satisfies FeatureFlagDefinition,
  driver_notification_actions_enabled: {
    key: "driver_notification_actions_enabled",
    source: "env",
    enabled: envEnabled("EXPO_PUBLIC_ENABLE_DRIVER_NOTIFICATION_ACTIONS"),
    description: "Enable quick notification actions and channels bootstrap.",
  } satisfies FeatureFlagDefinition,
  driver_messages_hub_enabled: {
    key: "driver_messages_hub_enabled",
    source: "env",
    enabled:
      process.env.EXPO_PUBLIC_ENABLE_DRIVER_MESSAGES_HUB === undefined
        ? true
        : envEnabled("EXPO_PUBLIC_ENABLE_DRIVER_MESSAGES_HUB"),
    description:
      "Hub Messages chauffeur (inbox mission-first, conversation enrichie, urgence dispatch).",
  } satisfies FeatureFlagDefinition,
  driver_mission_bar_enabled: {
    key: "driver_mission_bar_enabled",
    source: "env",
    enabled: envEnabled("EXPO_PUBLIC_ENABLE_DRIVER_MISSION_BAR"),
    description: "Enable driver mission bar runtime integration.",
  } satisfies FeatureFlagDefinition,
  realtime_auth_reconnect_enabled: {
    key: "realtime_auth_reconnect_enabled",
    source: "env",
    enabled: envEnabled("EXPO_PUBLIC_ENABLE_REALTIME_AUTH_RECONNECT"),
    description: "Enable auth-aware realtime reconnect policy with explicit resume orchestration.",
  } satisfies FeatureFlagDefinition,
  realtime_auth_exhaustion_guard_enabled: {
    key: "realtime_auth_exhaustion_guard_enabled",
    source: "env",
    enabled: envEnabled("EXPO_PUBLIC_ENABLE_REALTIME_AUTH_EXHAUSTION_GUARD"),
    description: "Expose explicit UI/session guardrails when realtime auth retries are exhausted.",
  } satisfies FeatureFlagDefinition,
  realtime_resync_transition_gate_enabled: {
    key: "realtime_resync_transition_gate_enabled",
    source: "env",
    enabled: envEnabled("EXPO_PUBLIC_ENABLE_REALTIME_RESYNC_TRANSITION_GATE"),
    description: "Trigger reconnect resync only on true connection transitions and gate bursty re-syncs.",
  } satisfies FeatureFlagDefinition,
  realtime_reconnect_circuit_breaker_enabled: {
    key: "realtime_reconnect_circuit_breaker_enabled",
    source: "env",
    enabled: envEnabled("EXPO_PUBLIC_ENABLE_REALTIME_RECONNECT_CIRCUIT_BREAKER"),
    description:
      "Active un cooldown reconnexion Socket.IO après rafales d'échecs (fenêtre glissante) pour limiter storms et consommation batterie.",
  } satisfies FeatureFlagDefinition,
  company_gps_flush_priority_lanes_enabled: {
    key: "company_gps_flush_priority_lanes_enabled",
    source: "env",
    enabled: envEnabled("EXPO_PUBLIC_ENABLE_COMPANY_GPS_FLUSH_PRIORITY_LANES"),
    description:
      "Active les lanes de flush GPS (critical/visible/background). Désactivé : pipeline P0 (critical immédiat, reste en batch 300ms).",
  } satisfies FeatureFlagDefinition,
  realtime_adaptive_polling_enabled: {
    key: "realtime_adaptive_polling_enabled",
    source: "env",
    enabled: envEnabled("EXPO_PUBLIC_ENABLE_REALTIME_ADAPTIVE_POLLING"),
    description: "Adapt full mission polling cadence based on realtime transport health.",
  } satisfies FeatureFlagDefinition,
  driver_network_idle_gating_enabled: {
    key: "driver_network_idle_gating_enabled",
    source: "env",
    enabled: envEnabled("EXPO_PUBLIC_ENABLE_DRIVER_NETWORK_IDLE_GATING"),
    description: "Reduce mission polling/sync cadence when no mission is relevant.",
  } satisfies FeatureFlagDefinition,
  driver_background_network_gating_enabled: {
    key: "driver_background_network_gating_enabled",
    source: "env",
    enabled: envEnabled("EXPO_PUBLIC_ENABLE_DRIVER_BACKGROUND_NETWORK_GATING"),
    description: "Apply background/deep-idle network cadence with app-state hysteresis.",
  } satisfies FeatureFlagDefinition,
  driver_sync_poll_harmonization_enabled: {
    key: "driver_sync_poll_harmonization_enabled",
    source: "env",
    enabled: envEnabled("EXPO_PUBLIC_ENABLE_DRIVER_SYNC_POLL_HARMONIZATION"),
    description: "Avoid redundant sync workload between realtime polling and sync engine.",
  } satisfies FeatureFlagDefinition,
  driver_network_2g_cadence_enabled: {
    key: "driver_network_2g_cadence_enabled",
    source: "env",
    enabled: envEnabled("EXPO_PUBLIC_ENABLE_DRIVER_NETWORK_2G_CADENCE"),
    description: "Apply explicit 2G/3G cadence profile for driver runtime loops.",
  } satisfies FeatureFlagDefinition,
  driver_http_adaptive_timeout_enabled: {
    key: "driver_http_adaptive_timeout_enabled",
    source: "env",
    enabled: envEnabled("EXPO_PUBLIC_ENABLE_DRIVER_HTTP_ADAPTIVE_TIMEOUT"),
    description: "Adjust HTTP timeout based on active network profile for driver APIs.",
  } satisfies FeatureFlagDefinition,
  driver_http_circuit_breaker_enabled: {
    key: "driver_http_circuit_breaker_enabled",
    source: "env",
    enabled: envEnabled("EXPO_PUBLIC_ENABLE_DRIVER_HTTP_CIRCUIT_BREAKER"),
    description: "Enable scoped HTTP circuit breakers for driver transport families.",
  } satisfies FeatureFlagDefinition,
  driver_transition_window_tuning_enabled: {
    key: "driver_transition_window_tuning_enabled",
    source: "env",
    enabled: envEnabled("EXPO_PUBLIC_ENABLE_DRIVER_TRANSITION_WINDOW_TUNING"),
    description: "Enable configurable replay window and replay attempt limits for mission transitions.",
  } satisfies FeatureFlagDefinition,
  driver_push_enabled: {
    key: "driver_push_enabled",
    source: "env",
    enabled: envEnabled("EXPO_PUBLIC_ENABLE_DRIVER_PUSH"),
    description: "Enable push registration and push listeners for drivers.",
  } satisfies FeatureFlagDefinition,
  company_push_enabled: {
    key: "company_push_enabled",
    source: "env",
    enabled: envEnabled("EXPO_PUBLIC_ENABLE_COMPANY_PUSH"),
    description: "Enable company push token registration and notification handlers.",
  } satisfies FeatureFlagDefinition,
  fleet_map_safe_markers: {
    key: "fleet_map_safe_markers",
    source: "env",
    enabled: envEnabled("EXPO_PUBLIC_FLEET_MAP_SAFE_MARKERS"),
    description:
      "Carte flotte : force les marqueurs en data-URI (sans PNG Metro/OTA). Rollback d'urgence via bootstrap feature_flags ou EXPO_PUBLIC_FLEET_MAP_SAFE_MARKERS=1.",
  } satisfies FeatureFlagDefinition,
  company_dispatch_enabled: {
    key: "company_dispatch_enabled",
    source: "env",
    enabled: envExplicitlyDisabled("EXPO_PUBLIC_ENABLE_COMPANY_DISPATCH")
      ? false
      : envEnabled("EXPO_PUBLIC_ENABLE_COMPANY_DISPATCH") ||
        (typeof __DEV__ !== "undefined" && __DEV__),
    description:
      "Déverrouille companyRealtimeBridge.connect dans (company)/_layout. Local dev: actif par défaut en __DEV__ (ou EXPO_PUBLIC_ENABLE_COMPANY_DISPATCH=1). Prod: désactivé sauf override session (feature_flags). Requiert aussi company_realtime + URL socket.",
  } satisfies FeatureFlagDefinition,
  mobile_context_cache_parking_enabled: {
    key: "mobile_context_cache_parking_enabled",
    source: "env",
    enabled:
      process.env.EXPO_PUBLIC_MOBILE_CONTEXT_CACHE_PARKING === undefined
        ? true
        : envEnabled("EXPO_PUBLIC_MOBILE_CONTEXT_CACHE_PARKING"),
    description:
      "Conserve le cache React Query par contexte (LRU 2) au switch entreprise/chauffeur au lieu de purge systématique.",
  } satisfies FeatureFlagDefinition,
  company_realtime_enabled: {
    key: "company_realtime_enabled",
    source: "env",
    enabled: envExplicitlyDisabled("EXPO_PUBLIC_ENABLE_COMPANY_REALTIME")
      ? false
      : envEnabled("EXPO_PUBLIC_ENABLE_COMPANY_REALTIME"),
    description:
      "Kill-switch for the company socket bridge. EXPO_PUBLIC_ENABLE_COMPANY_REALTIME=1 + URL (EXPO_PUBLIC_COMPANY_SOCKET_URL ou origine EXPO_PUBLIC_API_BASE_URL ; pas de repli DRIVER). En __DEV__, activé automatiquement si le dispatch company est actif (env ou feature_flags session). Voir expo-company-env.template.",
  } satisfies FeatureFlagDefinition,
  company_dispatch_screen_enabled: {
    key: "company_dispatch_screen_enabled",
    source: "env",
    enabled: envEnabled("EXPO_PUBLIC_ENABLE_COMPANY_DISPATCH_SCREEN"),
    description:
      "When true, dashboard CTA in semi-auto opens the propositions (dispatch) screen; otherwise repli vers courses.",
  } satisfies FeatureFlagDefinition,
  company_mobile_structured_ride_payload_enabled: {
    key: "company_mobile_structured_ride_payload_enabled",
    source: "env",
    enabled: envEnabled("EXPO_PUBLIC_ENABLE_COMPANY_MOBILE_STRUCTURED_RIDE_PAYLOAD"),
    description: "Enable structured mobile ride payload (label/place_id/lat/lon) for create and edit flows.",
  } satisfies FeatureFlagDefinition,
  company_mobile_chat_socket_enabled: {
    key: "company_mobile_chat_socket_enabled",
    source: "env",
    enabled: envEnabled("EXPO_PUBLIC_ENABLE_COMPANY_MOBILE_CHAT_SOCKET"),
    description: "Enable realtime socket consumption for company chat updates.",
  } satisfies FeatureFlagDefinition,
  company_mobile_map_clustering_enabled: {
    key: "company_mobile_map_clustering_enabled",
    source: "env",
    enabled: envEnabled("EXPO_PUBLIC_ENABLE_COMPANY_MOBILE_MAP_CLUSTERING"),
    description: "Enable clustering mode on company live drivers map.",
  } satisfies FeatureFlagDefinition,
  company_mobile_assign_optimistic_enabled: {
    key: "company_mobile_assign_optimistic_enabled",
    source: "env",
    enabled: envEnabled("EXPO_PUBLIC_ENABLE_COMPANY_MOBILE_ASSIGN_OPTIMISTIC"),
    description: "Enable optimistic assign/reassign updates in company mission lists.",
  } satisfies FeatureFlagDefinition,
  company_mobile_clients_readonly_enabled: {
    key: "company_mobile_clients_readonly_enabled",
    source: "env",
    enabled:
      envEnabled("EXPO_PUBLIC_ENABLE_COMPANY_MOBILE_CLIENTS_READONLY") ||
      (typeof __DEV__ !== "undefined" && __DEV__),
    description:
      "Lecture seule clients (company). Dev : actif par défaut (__DEV__). Prod : EXPO_PUBLIC_ENABLE_COMPANY_MOBILE_CLIENTS_READONLY=1 ou feature_flags session.",
  } satisfies FeatureFlagDefinition,
  company_mobile_invoices_readonly_enabled: {
    key: "company_mobile_invoices_readonly_enabled",
    source: "env",
    enabled:
      envEnabled("EXPO_PUBLIC_ENABLE_COMPANY_MOBILE_INVOICES_READONLY") ||
      (typeof __DEV__ !== "undefined" && __DEV__),
    description:
      "Lecture seule factures (company). Dev : actif par défaut (__DEV__). Prod : EXPO_PUBLIC_ENABLE_COMPANY_MOBILE_INVOICES_READONLY=1 ou feature_flags session.",
  } satisfies FeatureFlagDefinition,
  company_mobile_role_guards_enabled: {
    key: "company_mobile_role_guards_enabled",
    source: "env",
    enabled: envEnabled("EXPO_PUBLIC_ENABLE_COMPANY_MOBILE_ROLE_GUARDS"),
    description: "Enable additional role-based guards on sensitive company actions.",
  } satisfies FeatureFlagDefinition,
  perf_instrumentation_enabled: {
    key: "perf_instrumentation_enabled",
    source: "env",
    enabled:
      envEnabled("EXPO_PUBLIC_PERF_INSTRUMENTATION") ||
      (typeof __DEV__ !== "undefined" && __DEV__),
    description:
      "Instrumentation perf Phase 0 (notify, invalidate, JS long tasks, heap). Actif en __DEV__ ou EXPO_PUBLIC_PERF_INSTRUMENTATION=1.",
  } satisfies FeatureFlagDefinition,
  perf_chat_local_patch_enabled: {
    key: "perf_chat_local_patch_enabled",
    source: "env",
    enabled:
      envEnabled("EXPO_PUBLIC_PERF_CHAT_LOCAL_PATCH") ||
      (typeof __DEV__ !== "undefined" && __DEV__),
    description:
      "Phase A chat : patch React Query local au lieu d'invalidate. __DEV__ on, prod off sauf EXPO_PUBLIC_PERF_CHAT_LOCAL_PATCH=1.",
  } satisfies FeatureFlagDefinition,
  ios_startup_fatal_recovery_disabled: {
    key: "ios_startup_fatal_recovery_disabled",
    source: "external",
    enabled: false,
    description:
      "Kill-switch backend : désarme ErrorRecovery.crash() au startup iOS. Lu via bootstrap/version-check après hotfix natif. IOS_STARTUP_FATAL_RECOVERY_DISABLED=true côté serveur.",
  } satisfies FeatureFlagDefinition,
} as const;

export type FeatureFlagKey = keyof typeof featureFlags;
type RuntimeFlagValue = boolean | number | string | null;

let runtimeOverrides: Partial<Record<FeatureFlagKey, RuntimeFlagValue>> = {};
let runtimeFlagsVersion: string | null = null;

function toBoolean(value: RuntimeFlagValue | undefined): boolean | null {
  if (typeof value === "boolean") return value;
  if (typeof value === "number") return value !== 0;
  if (typeof value === "string") {
    const normalized = value.trim().toLowerCase();
    if (normalized === "1" || normalized === "true" || normalized === "yes" || normalized === "on") {
      return true;
    }
    if (normalized === "0" || normalized === "false" || normalized === "no" || normalized === "off") {
      return false;
    }
  }
  return null;
}

export function setRuntimeFeatureFlagOverrides(
  flags: Record<string, RuntimeFlagValue> | null | undefined
) {
  if (!flags) {
    runtimeOverrides = {};
    runtimeFlagsVersion = null;
    return;
  }
  const next: Partial<Record<FeatureFlagKey, RuntimeFlagValue>> = {};
  Object.keys(featureFlags).forEach((key) => {
    const typedKey = key as FeatureFlagKey;
    if (Object.prototype.hasOwnProperty.call(flags, typedKey)) {
      next[typedKey] = flags[typedKey];
    }
  });
  runtimeOverrides = next;
  const maybeVersion = flags.feature_flags_version;
  runtimeFlagsVersion =
    typeof maybeVersion === "string" && maybeVersion.trim().length > 0 ? maybeVersion : null;
}

function isCompanyDispatchEnabledResolved(): boolean {
  const runtimeDispatch = toBoolean(runtimeOverrides.company_dispatch_enabled);
  if (runtimeDispatch !== null) return runtimeDispatch;
  return featureFlags.company_dispatch_enabled.enabled;
}

export function isFeatureEnabled(key: FeatureFlagKey): boolean {
  const runtimeValue = toBoolean(runtimeOverrides[key]);
  if (runtimeValue !== null) return runtimeValue;
  if (
    key === "company_realtime_enabled" &&
    !envExplicitlyDisabled("EXPO_PUBLIC_ENABLE_COMPANY_REALTIME") &&
    typeof __DEV__ !== "undefined" &&
    __DEV__ &&
    isCompanyDispatchEnabledResolved()
  ) {
    return true;
  }
  return featureFlags[key].enabled;
}

/** Socket company attendu (dispatch + realtime) — pilote cockpit et _layout. */
export function isCompanyRealtimeSocketExpected(): boolean {
  return (
    isFeatureEnabled("company_dispatch_enabled") &&
    isFeatureEnabled("company_realtime_enabled")
  );
}

export function getFeatureFlagSource(key: FeatureFlagKey): FeatureFlagSource {
  if (runtimeOverrides[key] !== undefined) return "external";
  return featureFlags[key].source;
}

export function getRuntimeFlagsVersion(): string | null {
  return runtimeFlagsVersion;
}
