import { describe, expect, it, beforeEach } from "@jest/globals";
import { QueryClient } from "@tanstack/react-query";
import {
  applyContextCachePolicyOnSwitch,
  getParkedContextCount,
  restoreContextCache,
} from "./contextCache";
import {
  getDuplicateSocketEventCount,
  getSocketConnectionCount,
  resetPerfSocketSession,
} from "../observability/perfKpi";

/**
 * Gate Sprint 1.5 — simulation 20× switch (sans socket réel).
 * Reconnect ≤ 25 : observé en E2E device ; ici on valide cache LRU + gauges.
 */
describe("context switch stress (20 cycles)", () => {
  beforeEach(() => {
    resetPerfSocketSession();
  });

  it("keeps cached_context_count ≤ 2 and no duplicate socket events", () => {
    const queryClient = new QueryClient();
    const contexts = ["company:1", "driver:42"] as const;
    let maxParkedContexts = 0;

    for (let i = 0; i < 20; i += 1) {
      const from = contexts[i % 2];
      const to = contexts[(i + 1) % 2];
      queryClient.setQueryData(["ctx", to, "screen"], { cycle: i });
      applyContextCachePolicyOnSwitch(queryClient, from);
      restoreContextCache(queryClient, to);
      maxParkedContexts = Math.max(maxParkedContexts, getParkedContextCount());
    }

    expect(maxParkedContexts).toBeLessThanOrEqual(2);
    expect(getSocketConnectionCount()).toBeLessThanOrEqual(2);
    expect(getDuplicateSocketEventCount()).toBe(0);
  });
});
