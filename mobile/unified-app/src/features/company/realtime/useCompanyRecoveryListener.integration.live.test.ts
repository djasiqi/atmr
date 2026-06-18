/**
 * Phase 2 mobile recovery — D3.2 INTEGRATION live test.
 *
 * Cible : vérifier le throttle 30s du hook `useCompanyRecoveryListener`
 * sous spam d'events `company_socket_reconnected` / `company_data_stale_resync`.
 *
 * Activé uniquement quand RUN_LIVE_RECOVERY=1 est défini, sinon skip pour ne
 * pas casser la CI Jest standard.
 */

import { afterEach, beforeEach, describe, expect, it, jest } from "@jest/globals";

const RUN_LIVE = process.env.RUN_LIVE_RECOVERY === "1";
const describeLive = RUN_LIVE ? describe : describe.skip;

// Mock minimal Sentry pour éviter l'init native.
jest.mock("@sentry/react-native", () => ({
  setTag: jest.fn(),
  addBreadcrumb: jest.fn(),
}));

// Mock perf instrumentation — on capte la trigger string pour audit.
jest.mock("../../../core/observability/perfInstrumentation", () => ({
  __esModule: true,
  traceInvalidateQueries: jest.fn(
    async (_key: unknown, _trigger: string, fn: () => Promise<void>) => {
      await fn();
    }
  ),
  recordRealtimeNotify: jest.fn(),
  recordSocketEventByChannel: jest.fn(),
}));

// eslint-disable-next-line @typescript-eslint/no-require-imports
const React = require("react");
// eslint-disable-next-line @typescript-eslint/no-require-imports
const TestRenderer = require("react-test-renderer");
// eslint-disable-next-line @typescript-eslint/no-require-imports
const { QueryClient, QueryClientProvider } = require("@tanstack/react-query");
 
// eslint-disable-next-line @typescript-eslint/no-require-imports
const { contextRealtimeRouter } = require("../../../core/realtime/contextRealtimeRouter");
// eslint-disable-next-line @typescript-eslint/no-require-imports
const { useCompanyRecoveryListener, RECOVERY_THROTTLE_MS } = require("./useCompanyRecoveryListener");
// eslint-disable-next-line @typescript-eslint/no-require-imports
const { getRealtimeMetricsSnapshot, resetRealtimeMetricsForTests } = require("../../../core/observability/realtimeMetrics");

function HookHarness({
  contextId,
}: {
  contextId: string | null;
}): null {
  useCompanyRecoveryListener(contextId);
  return null;
}

function mountHookWithContext(contextId: string | null): {
  queryClient: InstanceType<typeof QueryClient>;
  invalidateSpy: ReturnType<typeof jest.spyOn>;
  unmount: () => void;
} {
  const queryClient = new QueryClient();
  const invalidateSpy = jest.spyOn(queryClient, "invalidateQueries");
  let renderer!: ReturnType<typeof TestRenderer.create>;
  TestRenderer.act(() => {
    renderer = TestRenderer.create(
      React.createElement(
        QueryClientProvider,
        { client: queryClient },
        React.createElement(HookHarness, { contextId })
      )
    );
  });
  return {
    queryClient,
    invalidateSpy,
    unmount: () => {
      TestRenderer.act(() => {
        renderer.unmount();
      });
    },
  };
}

describeLive("D3.2 mobile recovery — hook integration with throttle", () => {
  beforeEach(() => {
    resetRealtimeMetricsForTests();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it("invalidates 5 keys on first reconnect event", () => {
    const { invalidateSpy, unmount } = mountHookWithContext("company:42");
    TestRenderer.act(() => {
      contextRealtimeRouter.dispatch("company:42", {
        event_type: "company_socket_reconnected",
        context_type: "company",
      });
    });
    expect(invalidateSpy).toHaveBeenCalledTimes(5);
    expect(getRealtimeMetricsSnapshot().recoveryResyncTotal).toBe(1);
    expect(getRealtimeMetricsSnapshot().recoveryResyncByTrigger.reconnect).toBe(1);
    unmount();
  });

  it("throttles spam: 10x reconnect in 100ms → 1 resync only", () => {
    const { invalidateSpy, unmount } = mountHookWithContext("company:43");
    TestRenderer.act(() => {
      for (let i = 0; i < 10; i += 1) {
        contextRealtimeRouter.dispatch("company:43", {
          event_type: "company_socket_reconnected",
          context_type: "company",
        });
      }
    });
    expect(invalidateSpy).toHaveBeenCalledTimes(5);
    expect(getRealtimeMetricsSnapshot().recoveryResyncTotal).toBe(1);
    unmount();
  });

  it("mixes stale + reconnect within throttle window: only first wins", () => {
    const { invalidateSpy, unmount } = mountHookWithContext("company:44");
    TestRenderer.act(() => {
      contextRealtimeRouter.dispatch("company:44", {
        event_type: "company_data_stale_resync",
        context_type: "company",
      });
      contextRealtimeRouter.dispatch("company:44", {
        event_type: "company_socket_reconnected",
        context_type: "company",
      });
      contextRealtimeRouter.dispatch("company:44", {
        event_type: "company_data_stale_resync",
        context_type: "company",
      });
    });
    expect(invalidateSpy).toHaveBeenCalledTimes(5);
    const snap = getRealtimeMetricsSnapshot();
    expect(snap.recoveryResyncTotal).toBe(1);
    expect(snap.recoveryResyncByTrigger.stale).toBe(1);
    expect(snap.recoveryResyncByTrigger.reconnect).toBe(0);
    unmount();
  });

  it(
    "allows a second resync after throttle window elapses",
    () => {
      jest.useFakeTimers({ doNotFake: ["nextTick", "setImmediate"] });
      const baseNow = Date.now();
      const dateSpy = jest.spyOn(Date, "now").mockReturnValue(baseNow);

      const { invalidateSpy, unmount } = mountHookWithContext("company:45");
      TestRenderer.act(() => {
        contextRealtimeRouter.dispatch("company:45", {
          event_type: "company_socket_reconnected",
          context_type: "company",
        });
      });
      expect(invalidateSpy).toHaveBeenCalledTimes(5);

      dateSpy.mockReturnValue(baseNow + RECOVERY_THROTTLE_MS + 1_000);
      TestRenderer.act(() => {
        contextRealtimeRouter.dispatch("company:45", {
          event_type: "company_data_stale_resync",
          context_type: "company",
        });
      });

      expect(invalidateSpy).toHaveBeenCalledTimes(10);
      const snap = getRealtimeMetricsSnapshot();
      expect(snap.recoveryResyncTotal).toBe(2);
      expect(snap.recoveryResyncByTrigger.reconnect).toBe(1);
      expect(snap.recoveryResyncByTrigger.stale).toBe(1);

      dateSpy.mockRestore();
      unmount();
    },
    15_000
  );

  it("ignores non-recovery events", () => {
    const { invalidateSpy, unmount } = mountHookWithContext("company:46");
    TestRenderer.act(() => {
      contextRealtimeRouter.dispatch("company:46", {
        event_type: "booking_updated",
        context_type: "company",
      });
      contextRealtimeRouter.dispatch("company:46", {
        event_type: "dispatch_assignment",
        context_type: "company",
      });
    });
    expect(invalidateSpy).not.toHaveBeenCalled();
    expect(getRealtimeMetricsSnapshot().recoveryResyncTotal).toBe(0);
    unmount();
  });

  it("does not subscribe when contextId is null", () => {
    const { invalidateSpy, unmount } = mountHookWithContext(null);
    TestRenderer.act(() => {
      contextRealtimeRouter.dispatch("company:47", {
        event_type: "company_socket_reconnected",
        context_type: "company",
      });
    });
    expect(invalidateSpy).not.toHaveBeenCalled();
    unmount();
  });
});

if (!RUN_LIVE) {
  // Garantit qu'au moins un test existe dans le fichier pour Jest.
  describe("D3.2 live (skipped)", () => {
    it("requires RUN_LIVE_RECOVERY=1", () => {
      expect(RUN_LIVE).toBe(false);
    });
  });
}
