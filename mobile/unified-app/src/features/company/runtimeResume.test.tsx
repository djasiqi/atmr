import React from "react";
import { beforeEach, describe, expect, it, jest } from "@jest/globals";
import { act, create } from "react-test-renderer";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useCompanyRuntimeResume } from "./runtimeResume";

const mockAddEventListener = jest.fn();
const mockRemoveAppStateListener = jest.fn();
let appStateCallback: ((state: "active" | "inactive" | "background") => void) | null = null;

const mockBootstrapSession = jest.fn();
const mockRefreshAuthTokenSingleflight = jest.fn();
const mockSetResumeAttemptCorrelationId = jest.fn();
const mockAppendSessionJournalEvent = jest.fn();
const mockBridgeConnect = jest.fn();
const mockBridgeReconnect = jest.fn();
const mockBridgeGetSnapshot = jest.fn();
const mockPerformCompanyRecoveryResync = jest.fn();

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

jest.mock("../../core/sessionProvider", () => ({
  useSession: () => ({
    status: "ready",
    bootstrapSession: () => mockBootstrapSession(),
  }),
}));

jest.mock("../../core/api/client", () => ({
  setResumeAttemptCorrelationId: (value: string | null) =>
    mockSetResumeAttemptCorrelationId(value),
}));

jest.mock("../../core/auth/authTokenOrchestrator", () => ({
  refreshAuthTokenSingleflight: (reason: string) => mockRefreshAuthTokenSingleflight(reason),
}));

jest.mock("../../core/observability/sessionJournal", () => ({
  appendSessionJournalEvent: (...args: unknown[]) => mockAppendSessionJournalEvent(...args),
}));

jest.mock("./realtime/companyRealtimeBridge", () => ({
  companyRealtimeBridge: {
    connect: (...args: unknown[]) => mockBridgeConnect(...args),
    reconnect: () => mockBridgeReconnect(),
    getSnapshot: () => mockBridgeGetSnapshot(),
  },
}));

jest.mock("./realtime/useCompanyRecoveryListener", () => ({
  performCompanyRecoveryResync: (...args: unknown[]) => mockPerformCompanyRecoveryResync(...args),
}));

function RuntimeResumeHarness(props: { contextId: string | null; enabled: boolean }) {
  useCompanyRuntimeResume(props);
  return null;
}

describe("useCompanyRuntimeResume", () => {
  beforeEach(() => {
    appStateCallback = null;
    mockAddEventListener.mockReset();
    mockRemoveAppStateListener.mockReset();
    mockBootstrapSession.mockReset();
    mockRefreshAuthTokenSingleflight.mockReset();
    mockSetResumeAttemptCorrelationId.mockReset();
    mockAppendSessionJournalEvent.mockReset();
    mockBridgeConnect.mockReset();
    mockBridgeReconnect.mockReset();
    mockBridgeGetSnapshot.mockReset();
    mockPerformCompanyRecoveryResync.mockReset();

    mockRefreshAuthTokenSingleflight.mockResolvedValue(true);
    mockBridgeGetSnapshot.mockReturnValue({
      status: "failed",
      contextId: "company:99",
      connected: false,
    });
  });

  it("runs refresh + bridge reconnect + recovery resync on foreground", async () => {
    const queryClient = new QueryClient();
    let renderer: ReturnType<typeof create>;
    await act(async () => {
      renderer = create(
        <QueryClientProvider client={queryClient}>
          <RuntimeResumeHarness contextId="company:99" enabled />
        </QueryClientProvider>
      );
    });

    await act(async () => {
      appStateCallback?.("inactive");
      appStateCallback?.("active");
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(mockRefreshAuthTokenSingleflight).toHaveBeenCalledWith("company_foreground_resume");
    expect(mockBridgeReconnect).toHaveBeenCalled();
    expect(mockPerformCompanyRecoveryResync).toHaveBeenCalledWith(
      queryClient,
      "company:99",
      "reconnect"
    );
    expect(mockAppendSessionJournalEvent).toHaveBeenCalledWith(
      "session.company.resume.start",
      expect.objectContaining({ resume_attempt_id: expect.any(String) }),
      "company:99"
    );
    expect(mockAppendSessionJournalEvent).toHaveBeenCalledWith(
      "session.company.resume.success",
      expect.any(Object),
      "company:99"
    );

    await act(async () => {
      renderer!.unmount();
    });
  });

  it("does not subscribe when disabled", async () => {
    const queryClient = new QueryClient();
    await act(async () => {
      create(
        <QueryClientProvider client={queryClient}>
          <RuntimeResumeHarness contextId="company:99" enabled={false} />
        </QueryClientProvider>
      );
    });
    expect(mockAddEventListener).not.toHaveBeenCalled();
  });
});
