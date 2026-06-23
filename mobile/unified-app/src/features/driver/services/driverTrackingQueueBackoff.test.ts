import {
  resolveQueueSuspendMs,
} from "./driverTrackingQueueBackoff";
import type { TrackingSendErrorMeta } from "./driverTrackingSendErrorFormat";

describe("driverTrackingQueueBackoff", () => {
  it("suspend 429 avec Retry-After", () => {
    const meta: TrackingSendErrorMeta = {
      error_class: "http",
      error_message: "rate limit",
      http_status: 429,
      api_error_code: null,
      transport_code: null,
      retry_after_seconds: 30,
    };
    const plan = resolveQueueSuspendMs(meta, meta.retry_after_seconds);
    expect(plan).toEqual({ suspendMs: 30_000, reason: "rate_limit" });
  });

  it("suspend circuit breaker", () => {
    const meta: TrackingSendErrorMeta = {
      error_class: "circuit_open",
      error_message: "open",
      http_status: null,
      api_error_code: "HTTP_CIRCUIT_BREAKER_OPEN",
      transport_code: null,
      retry_after_seconds: null,
    };
    const plan = resolveQueueSuspendMs(meta, null);
    expect(plan?.reason).toBe("circuit_breaker");
  });

  it("pas de suspension pour erreur generique", () => {
    const meta: TrackingSendErrorMeta = {
      error_class: "network",
      error_message: "offline",
      http_status: null,
      api_error_code: null,
      transport_code: null,
      retry_after_seconds: null,
    };
    expect(resolveQueueSuspendMs(meta, null)).toBeNull();
  });
});
