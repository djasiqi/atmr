import React from "react";
import { beforeEach, describe, expect, it, jest } from "@jest/globals";
import { act, create } from "react-test-renderer";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useDriverRuntimeResume } from "./runtimeResume";

const mockAddEventListener = jest.fn();
const mockRemoveAppStateListener = jest.fn();
let appStateCallback: ((state: "active" | "inactive" | "background") => void) | null = null;

const mockEmitTelemetry = jest.fn();
const mockBootstrapSession = jest.fn();
const mockRefreshAuthTokenNow = jest.fn();
const mockSetResumeAttemptCorrelationId = jest.fn();
const mockRealtimeConnect = jest.fn();
const mockReconcileDriverMissions = jest.fn();
const mockOfflineFlush = jest.fn();
const mockIsFeatureEnabled = jest.fn();

jest.mock("react-native", () => ({
  AppState: {
    currentState: "active",
    addEventListener: (event: string, callback: (next: "active" | "inactive" | "background") => void) => {
      mockAddEventListener(event, callback);
      appStateCallback = callback;
      return { remove: mockRemoveAppStateListener };
    },
  },
}));

jest.mock("../../core/observability/driverTelemetry", () => ({
  emitDriverTelemetry: (event: string, payload: unknown) => mockEmitTelemetry(event, payload),
}));

jest.mock("../../core/sessionProvider", () => ({
  useSession: () => ({
    status: "ready",
    bootstrapSession: () => mockBootstrapSession(),
  }),
}));

jest.mock("../../core/api/client", () => ({
  refreshAuthTokenNow: () => mockRefreshAuthTokenNow(),
  setResumeAttemptCorrelationId: (value: string | null) =>
    mockSetResumeAttemptCorrelationId(value),
}));

jest.mock("../../core/realtime/realtimeManager", () => ({
  realtimeManager: {
    connect: (...args: unknown[]) => mockRealtimeConnect(...args),
  },
}));

jest.mock("../../core/featureFlags/registry", () => ({
  isFeatureEnabled: (flag: string) => mockIsFeatureEnabled(flag),
}));

jest.mock("./sync", () => ({
  reconcileDriverMissions: (...args: unknown[]) => mockReconcileDriverMissions(...args),
}));

jest.mock("./offlineQueue", () => ({
  driverOfflineQueue: {
    flush: () => mockOfflineFlush(),
  },
}));

function RuntimeResumeHarness(props: {
  contextId: string | null;
  enabled: boolean;
  onForegroundResume?: () => Promise<void>;
}) {
  useDriverRuntimeResume(props);
  return null;
}

function getResumeEvents() {
  return mockEmitTelemetry.mock.calls
    .map(([event, payload]) => ({ event, payload }))
    .filter(({ event }) => String(event).startsWith("driver.runtime.resume."));
}

describe("useDriverRuntimeResume", () => {
  beforeEach(() => {
    appStateCallback = null;
    mockAddEventListener.mockReset();
    mockRemoveAppStateListener.mockReset();
    mockEmitTelemetry.mockReset();
    mockBootstrapSession.mockReset();
    mockRefreshAuthTokenNow.mockReset();
    mockSetResumeAttemptCorrelationId.mockReset();
    mockRealtimeConnect.mockReset();
    mockReconcileDriverMissions.mockReset();
    mockOfflineFlush.mockReset();
    mockIsFeatureEnabled.mockReset();

    mockRefreshAuthTokenNow.mockResolvedValue(true);
    mockReconcileDriverMissions.mockResolvedValue({ missions: [], queue: { sent: 0, dropped: 0, failed: 0 } });
    mockOfflineFlush.mockResolvedValue({ sent: 0, dropped: 0, failed: 0 });
    mockIsFeatureEnabled.mockImplementation((flag: string) => {
      if (flag === "realtime_auth_reconnect_enabled") return true;
      if (flag === "realtime_socket_enabled") return true;
      return false;
    });
  });

  it("runs one full resume sequence on inactive -> active transition", async () => {
    const queryClient = new QueryClient();
    const onForegroundResume = jest.fn().mockResolvedValue(undefined);
    let renderer: ReturnType<typeof create>;
    await act(async () => {
      renderer = create(
        <QueryClientProvider client={queryClient}>
          <RuntimeResumeHarness
            contextId="driver:42"
            enabled
            onForegroundResume={onForegroundResume}
          />
        </QueryClientProvider>
      );
    });

    await act(async () => {
      appStateCallback?.("inactive");
      appStateCallback?.("active");
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(mockRefreshAuthTokenNow).toHaveBeenCalledTimes(1);
    expect(mockRealtimeConnect).toHaveBeenCalledWith("driver:42", { enableSocket: true });
    expect(mockReconcileDriverMissions).toHaveBeenCalledTimes(1);
    expect(mockOfflineFlush).toHaveBeenCalledTimes(1);
    expect(onForegroundResume).toHaveBeenCalledTimes(1);
    expect(mockEmitTelemetry).toHaveBeenCalledWith(
      "driver.runtime.resume.success",
      expect.objectContaining({ source: "driver.runtime.resume", context_id: "driver:42" })
    );
    const resumeEvents = getResumeEvents();
    expect(resumeEvents.map((entry) => entry.event)).toEqual([
      "driver.runtime.resume.start",
      "driver.runtime.resume.success",
    ]);
    const startAttemptId = String((resumeEvents[0]?.payload as any)?.resume_attempt_id ?? "");
    const successAttemptId = String((resumeEvents[1]?.payload as any)?.resume_attempt_id ?? "");
    expect(startAttemptId.length).toBeGreaterThan(0);
    expect(successAttemptId).toBe(startAttemptId);

    await act(async () => {
      renderer!.unmount();
    });
  });

  it("does not subscribe when disabled or context is null", async () => {
    const queryClient = new QueryClient();
    await act(async () => {
      create(
        <QueryClientProvider client={queryClient}>
          <RuntimeResumeHarness contextId={null} enabled={false} />
        </QueryClientProvider>
      );
    });
    expect(mockAddEventListener).not.toHaveBeenCalled();
  });

  it("emits failure telemetry when reconcile fails", async () => {
    const queryClient = new QueryClient();
    mockReconcileDriverMissions
      .mockRejectedValueOnce(new Error("sync_failed_a"))
      .mockRejectedValueOnce(new Error("sync_failed_b"));
    let renderer: ReturnType<typeof create>;
    await act(async () => {
      renderer = create(
        <QueryClientProvider client={queryClient}>
          <RuntimeResumeHarness contextId="driver:42" enabled />
        </QueryClientProvider>
      );
    });

    await act(async () => {
      appStateCallback?.("inactive");
      appStateCallback?.("active");
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(mockEmitTelemetry).toHaveBeenCalledWith(
      "driver.runtime.resume.failure",
      expect.objectContaining({
        reason: "sync_failed_a",
        context_id: "driver:42",
        retry_count: 1,
        will_retry: true,
      })
    );
    const resumeEvents = getResumeEvents();
    expect(resumeEvents.map((entry) => entry.event)).toEqual([
      "driver.runtime.resume.start",
      "driver.runtime.resume.failure",
      "driver.runtime.resume.failure",
    ]);
    const attemptIds = resumeEvents.map((entry) => (entry.payload as any)?.resume_attempt_id);
    expect(new Set(attemptIds).size).toBe(1);
    const lastFailure = resumeEvents[2]?.payload as any;
    expect(lastFailure?.reason).toBe("sync_failed_b");
    expect(lastFailure?.will_retry).toBe(false);

    await act(async () => {
      renderer!.unmount();
    });
  });

  it("prevents duplicate resume sequence when rapid events overlap", async () => {
    let releaseReconcile: (() => void) | null = null;
    mockReconcileDriverMissions.mockImplementation(
      () =>
        new Promise((resolve) => {
          releaseReconcile = () =>
            resolve({ missions: [], queue: { sent: 0, dropped: 0, failed: 0 } });
        })
    );
    const queryClient = new QueryClient();
    let renderer: ReturnType<typeof create>;
    await act(async () => {
      renderer = create(
        <QueryClientProvider client={queryClient}>
          <RuntimeResumeHarness contextId="driver:42" enabled />
        </QueryClientProvider>
      );
    });

    await act(async () => {
      appStateCallback?.("inactive");
      appStateCallback?.("active");
      appStateCallback?.("inactive");
      appStateCallback?.("active");
      await Promise.resolve();
    });

    expect(mockReconcileDriverMissions).toHaveBeenCalledTimes(1);
    await act(async () => {
      releaseReconcile?.();
      await Promise.resolve();
    });

    await act(async () => {
      renderer!.unmount();
    });
  });

  it("allows a new resume attempt after a previous failure", async () => {
    const queryClient = new QueryClient();
    mockReconcileDriverMissions
      .mockRejectedValueOnce(new Error("first_resume_failed_a"))
      .mockRejectedValueOnce(new Error("first_resume_failed_b"))
      .mockResolvedValueOnce({ missions: [], queue: { sent: 0, dropped: 0, failed: 0 } });

    let renderer: ReturnType<typeof create>;
    await act(async () => {
      renderer = create(
        <QueryClientProvider client={queryClient}>
          <RuntimeResumeHarness contextId="driver:42" enabled />
        </QueryClientProvider>
      );
    });

    await act(async () => {
      appStateCallback?.("inactive");
      appStateCallback?.("active");
      await Promise.resolve();
      await Promise.resolve();
    });

    await act(async () => {
      appStateCallback?.("inactive");
      appStateCallback?.("active");
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(mockReconcileDriverMissions).toHaveBeenCalledTimes(3);
    expect(mockEmitTelemetry).toHaveBeenCalledWith(
      "driver.runtime.resume.failure",
      expect.objectContaining({ reason: "first_resume_failed_a", will_retry: true })
    );
    expect(mockEmitTelemetry).toHaveBeenCalledWith(
      "driver.runtime.resume.failure",
      expect.objectContaining({ reason: "first_resume_failed_b", will_retry: false })
    );
    expect(mockEmitTelemetry).toHaveBeenCalledWith(
      "driver.runtime.resume.success",
      expect.objectContaining({ context_id: "driver:42" })
    );

    await act(async () => {
      renderer!.unmount();
    });
  });

  it("retries reconnect path when realtime connect throws once", async () => {
    const queryClient = new QueryClient();
    mockRealtimeConnect
      .mockImplementationOnce(() => {
        throw new Error("socket_connect_failed");
      })
      .mockImplementationOnce(() => undefined);

    let renderer: ReturnType<typeof create>;
    await act(async () => {
      renderer = create(
        <QueryClientProvider client={queryClient}>
          <RuntimeResumeHarness contextId="driver:42" enabled />
        </QueryClientProvider>
      );
    });

    await act(async () => {
      appStateCallback?.("inactive");
      appStateCallback?.("active");
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(mockRealtimeConnect).toHaveBeenCalledTimes(2);
    expect(mockEmitTelemetry).toHaveBeenCalledWith(
      "driver.runtime.resume.failure",
      expect.objectContaining({ reason: "socket_connect_failed", will_retry: true })
    );
    expect(mockEmitTelemetry).toHaveBeenCalledWith(
      "driver.runtime.resume.success",
      expect.objectContaining({ context_id: "driver:42", retry_count: 1 })
    );
    const resumeEvents = getResumeEvents();
    expect(resumeEvents.map((entry) => entry.event)).toEqual([
      "driver.runtime.resume.start",
      "driver.runtime.resume.failure",
      "driver.runtime.resume.success",
    ]);
    const successCount = resumeEvents.filter((entry) => entry.event === "driver.runtime.resume.success").length;
    expect(successCount).toBe(1);

    await act(async () => {
      renderer!.unmount();
    });
  });

  it("recovers on next resume when tracking flush callback fails", async () => {
    const queryClient = new QueryClient();
    const onForegroundResume = jest
      .fn()
      .mockRejectedValueOnce(new Error("flush_tracking_failed"))
      .mockResolvedValueOnce(undefined);
    let renderer: ReturnType<typeof create>;

    await act(async () => {
      renderer = create(
        <QueryClientProvider client={queryClient}>
          <RuntimeResumeHarness contextId="driver:42" enabled onForegroundResume={onForegroundResume} />
        </QueryClientProvider>
      );
    });

    await act(async () => {
      appStateCallback?.("inactive");
      appStateCallback?.("active");
      await Promise.resolve();
      await Promise.resolve();
    });

    await act(async () => {
      appStateCallback?.("inactive");
      appStateCallback?.("active");
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(onForegroundResume).toHaveBeenCalledTimes(3);
    expect(mockEmitTelemetry).toHaveBeenCalledWith(
      "driver.runtime.resume.failure",
      expect.objectContaining({ reason: "flush_tracking_failed" })
    );
    expect(mockEmitTelemetry).toHaveBeenCalledWith(
      "driver.runtime.resume.success",
      expect.objectContaining({ context_id: "driver:42" })
    );
    const resumeEvents = getResumeEvents();
    const attempts = new Map<string, string[]>();
    for (const entry of resumeEvents) {
      const attemptId = String((entry.payload as any)?.resume_attempt_id ?? "");
      if (!attempts.has(attemptId)) attempts.set(attemptId, []);
      attempts.get(attemptId)?.push(String(entry.event));
    }
    expect(attempts.size).toBe(2);
    const sequences = Array.from(attempts.values());
    expect(sequences[0]).toEqual([
      "driver.runtime.resume.start",
      "driver.runtime.resume.failure",
      "driver.runtime.resume.success",
    ]);
    expect(sequences[1]).toEqual([
      "driver.runtime.resume.start",
      "driver.runtime.resume.success",
    ]);

    await act(async () => {
      renderer!.unmount();
    });
  });
});
