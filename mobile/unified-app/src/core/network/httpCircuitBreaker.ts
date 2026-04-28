import { emitDriverTelemetry } from "../observability/driverTelemetry";
import { isFeatureEnabled } from "../featureFlags/registry";

export type HttpCircuitBreakerScope = "tracking_http" | "mission_transition" | "mission_sync";
type BreakerState = "closed" | "open" | "half_open";

type ScopeState = {
  state: BreakerState;
  failures: number;
  openedAtMs: number | null;
  halfOpenInFlight: boolean;
};

const DEFAULT_FAILURE_THRESHOLD = Number(
  process.env.EXPO_PUBLIC_DRIVER_HTTP_BREAKER_FAILURE_THRESHOLD ?? "3"
);
const DEFAULT_COOLDOWN_MS = Number(process.env.EXPO_PUBLIC_DRIVER_HTTP_BREAKER_COOLDOWN_MS ?? "20000");

const scopeStates = new Map<HttpCircuitBreakerScope, ScopeState>();

function getScopeState(scope: HttpCircuitBreakerScope): ScopeState {
  const existing = scopeStates.get(scope);
  if (existing) return existing;
  const fresh: ScopeState = {
    state: "closed",
    failures: 0,
    openedAtMs: null,
    halfOpenInFlight: false,
  };
  scopeStates.set(scope, fresh);
  return fresh;
}

function transition(scope: HttpCircuitBreakerScope, nextState: BreakerState, reason: string) {
  const state = getScopeState(scope);
  state.state = nextState;
  if (nextState === "open") {
    state.openedAtMs = Date.now();
  } else if (nextState === "closed") {
    state.openedAtMs = null;
    state.failures = 0;
    state.halfOpenInFlight = false;
  }
  emitDriverTelemetry("driver.http.circuit_breaker", {
    source: "core.network.httpCircuitBreaker",
    scope,
    state: nextState,
    reason,
    failures: state.failures,
  });
}

export function shouldAllowHttpRequest(scope: HttpCircuitBreakerScope): boolean {
  if (!isFeatureEnabled("driver_http_circuit_breaker_enabled")) {
    return true;
  }
  const state = getScopeState(scope);
  if (state.state === "closed") {
    return true;
  }
  if (state.state === "half_open") {
    if (state.halfOpenInFlight) {
      return false;
    }
    state.halfOpenInFlight = true;
    return true;
  }
  const openedAtMs = state.openedAtMs ?? 0;
  if (Date.now() - openedAtMs >= DEFAULT_COOLDOWN_MS) {
    transition(scope, "half_open", "cooldown_elapsed");
    state.halfOpenInFlight = true;
    return true;
  }
  return false;
}

export function onHttpRequestSuccess(scope: HttpCircuitBreakerScope) {
  if (!isFeatureEnabled("driver_http_circuit_breaker_enabled")) {
    return;
  }
  const state = getScopeState(scope);
  if (state.state === "half_open") {
    // Strict close rule: close only after real backend success.
    transition(scope, "closed", "backend_success");
    return;
  }
  state.failures = 0;
  state.halfOpenInFlight = false;
}

export function onHttpRequestFailure(scope: HttpCircuitBreakerScope, reason: string) {
  if (!isFeatureEnabled("driver_http_circuit_breaker_enabled")) {
    return;
  }
  const state = getScopeState(scope);
  state.halfOpenInFlight = false;
  state.failures += 1;
  if (state.state === "half_open") {
    transition(scope, "open", `half_open_failed:${reason}`);
    return;
  }
  if (state.failures >= DEFAULT_FAILURE_THRESHOLD) {
    transition(scope, "open", `threshold_reached:${reason}`);
  }
}

export function getHttpCircuitBreakerState(scope: HttpCircuitBreakerScope): BreakerState {
  return getScopeState(scope).state;
}
