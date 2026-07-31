import React from "react";
import { describe, expect, it, jest, beforeEach } from "@jest/globals";
import { act, create } from "react-test-renderer";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type {
  fetchBootstrap,
  hasAuthToken,
  hasStoredRefreshToken,
  login,
  logoutSession,
  refreshAuthTokenNow,
  switchContext,
} from "./api/client";
import { SessionProvider, useSession } from "./sessionProvider";

const mockFetchBootstrap = jest.fn() as jest.MockedFunction<typeof fetchBootstrap>;
const mockHasAuthToken = jest.fn() as jest.MockedFunction<typeof hasAuthToken>;
const mockHasStoredRefreshToken = jest.fn() as jest.MockedFunction<typeof hasStoredRefreshToken>;
const mockLogin = jest.fn() as jest.MockedFunction<typeof login>;
const mockLogoutSession = jest.fn() as jest.MockedFunction<typeof logoutSession>;
const mockRefreshAuthTokenNow = jest.fn() as jest.MockedFunction<typeof refreshAuthTokenNow>;
const mockSwitchContext = jest.fn() as jest.MockedFunction<typeof switchContext>;
const mockSetActiveContextIdForApi = jest.fn() as jest.MockedFunction<(contextId: string | null) => void>;
const mockOnContextSwitch = jest.fn() as jest.MockedFunction<(nextContextId: string | null) => void>;
const mockDisconnect = jest.fn() as jest.MockedFunction<() => void>;
const mockOnAuthExhausted = jest.fn().mockReturnValue(() => undefined);
const mockApplyContextCachePolicyOnSwitch = jest.fn() as jest.MockedFunction<
  (queryClient: QueryClient, contextId: string | null) => void
>;
const mockRestoreContextCache = jest.fn().mockReturnValue(false) as jest.MockedFunction<
  (queryClient: QueryClient, contextId: string | null) => boolean
>;
const mockClearAllContextCache = jest.fn() as jest.MockedFunction<(queryClient: QueryClient) => void>;
const mockPurgeDriverProfileCache = jest.fn() as jest.Mock<any>;
const mockAttemptRestRecovery = jest.fn(async () => "no_action");
const mockPerformExplicitLogout = jest.fn(async (params: {
  onLogoutClaimed?: (gen: number) => void;
  commitSessionStateIfCurrent: (gen: number) => boolean;
}) => {
  params.onLogoutClaimed?.(2);
  params.commitSessionStateIfCurrent(2);
  return { status: "completed" as const, logoutGeneration: 2, lifecycleOperationId: "op" };
});

jest.mock("./api/client", () => ({
  fetchBootstrap: (activeContextId?: string | null) => mockFetchBootstrap(activeContextId),
  hasAuthToken: () => mockHasAuthToken(),
  hasStoredRefreshToken: () => mockHasStoredRefreshToken(),
  login: (email: string, password: string) => mockLogin(email, password),
  logoutSession: () => mockLogoutSession(),
  refreshAuthTokenNow: () => mockRefreshAuthTokenNow(),
  switchContext: (targetContextId: string) => mockSwitchContext(targetContextId),
  setActiveContextIdForApi: (contextId: string | null) => mockSetActiveContextIdForApi(contextId),
  getLastRefreshErrorCode: () => null,
}));

jest.mock("./auth/authRecoveryCoordinator", () => ({
  attemptRestRecovery: (...args: unknown[]) => mockAttemptRestRecovery(...args),
  flushPendingRevocationTombstone: jest.fn(async () => true),
  performLogout: jest.fn(async () => undefined),
  performExplicitLogout: (...args: unknown[]) =>
    mockPerformExplicitLogout(...(args as [Parameters<typeof mockPerformExplicitLogout>[0]])),
  applyTerminalRevocationIfCurrent: jest.fn(async () => "stale"),
  finishInterruptedExplicitLogout: jest.fn(async () => undefined),
  persistOfflineSnapshot: jest.fn(async () => undefined),
  restoreOfflineSessionSnapshot: jest.fn(async () => ({ kind: "anonymous" })),
  hasPendingRevocationTombstone: jest.fn(async () => false),
}));

jest.mock("./auth/authCredentialStore", () => ({
  getSessionGenerationId: () => 1,
  isCurrentSessionGeneration: () => true,
  readSessionEnvelope: jest.fn(async () => ({
    status: "found",
    value: {
      session_id: "sess-test",
      device_installation_id: "dev",
      revocation_secret: "sec",
    },
  })),
}));

jest.mock("./realtime/realtimeManager", () => ({
  realtimeManager: {
    onContextSwitch: (nextContextId: string | null) => mockOnContextSwitch(nextContextId),
    disconnect: () => mockDisconnect(),
    onAuthExhausted: (
      cb: (reason: "exhausted" | "terminal", code?: string) => void
    ) => mockOnAuthExhausted(cb),
  },
}));

jest.mock("./cache/contextCache", () => ({
  applyContextCachePolicyOnSwitch: (queryClient: QueryClient, contextId: string | null) =>
    mockApplyContextCachePolicyOnSwitch(queryClient, contextId),
  restoreContextCache: (queryClient: QueryClient, contextId: string | null) =>
    mockRestoreContextCache(queryClient, contextId),
  clearAllContextCache: (queryClient: QueryClient) => mockClearAllContextCache(queryClient),
}));

jest.mock("./cache/prefetchContextTarget", () => ({
  prefetchContextTarget: jest.fn(),
}));

jest.mock("../features/driver/services/driverProfileCache", () => ({
  purgeDriverProfileCache: () => mockPurgeDriverProfileCache(),
}));

function buildBootstrap(activeContextId: string | null = "driver:42") {
  return {
    bootstrap_version: "1.0.0",
    is_authenticated: true,
    user: { id: "u-1", email: "driver@lirie.ch", full_name: "Driver Test" },
    account_status: "active" as const,
    onboarding_status: { required: false },
    available_contexts: [
      {
        context_id: "driver:42",
        context_type: "driver" as const,
        label: "Driver",
        permissions: ["mission:read", "mission:update_status"],
        is_default: activeContextId === "driver:42",
      },
      {
        context_id: "client:self",
        context_type: "client" as const,
        label: "Client",
        permissions: ["booking:read"],
        is_default: activeContextId === "client:self",
      },
    ],
    active_context_id: activeContextId,
    feature_flags: {},
    min_supported_app_version: "0.1.0",
    maintenance_mode: false,
    degraded_mode: false,
    server_time: new Date().toISOString(),
    request_id: "req-12345678",
  };
}

function buildUnauthenticatedBootstrap() {
  return {
    ...buildBootstrap(null),
    is_authenticated: false,
    available_contexts: [],
    active_context_id: null,
  };
}

type SessionHandle = ReturnType<typeof useSession>;

async function buildHarness(handle: { current: SessionHandle | null }) {
  const queryClient = new QueryClient();
  function Capture() {
    handle.current = useSession();
    return null;
  }
  let renderer!: ReturnType<typeof create>;
  await act(async () => {
    renderer = create(
      <QueryClientProvider client={queryClient}>
        <SessionProvider>
          <Capture />
        </SessionProvider>
      </QueryClientProvider>
    );
  });
  return { renderer, queryClient };
}

describe("session provider gates", () => {
  beforeEach(() => {
    mockFetchBootstrap.mockReset();
    mockHasAuthToken.mockReset();
    mockHasStoredRefreshToken.mockReset();
    mockLogin.mockReset();
    mockLogoutSession.mockReset();
    mockRefreshAuthTokenNow.mockReset();
    mockSwitchContext.mockReset();
    mockSetActiveContextIdForApi.mockReset();
    mockOnContextSwitch.mockReset();
    mockDisconnect.mockReset();
    mockApplyContextCachePolicyOnSwitch.mockReset();
    mockRestoreContextCache.mockReset();
    mockRestoreContextCache.mockReturnValue(false);
    mockClearAllContextCache.mockReset();
    mockPurgeDriverProfileCache.mockReset();
    mockAttemptRestRecovery.mockReset();
    mockAttemptRestRecovery.mockResolvedValue("no_action");
    mockPerformExplicitLogout.mockClear();
    mockPurgeDriverProfileCache.mockResolvedValue(undefined);
    mockHasAuthToken.mockReturnValue(false);
    mockHasStoredRefreshToken.mockResolvedValue(false);
    mockRefreshAuthTokenNow.mockResolvedValue(false);
  });

  it("handles cold start bootstrap and resolves active context", async () => {
    mockFetchBootstrap.mockResolvedValue(buildBootstrap("driver:42"));
    const handle: { current: SessionHandle | null } = { current: null };
    const { renderer } = await buildHarness(handle);

    await act(async () => {
      await handle.current?.bootstrapSession();
    });

    expect(mockFetchBootstrap).toHaveBeenCalledWith(null);
    expect(handle.current?.status).toBe("ready");
    expect(handle.current?.activeContext?.context_id).toBe("driver:42");
    expect(mockSetActiveContextIdForApi).toHaveBeenCalledWith("driver:42");
    expect(mockOnContextSwitch).toHaveBeenCalledWith("driver:42");
    await act(async () => {
      renderer.unmount();
    });
  });

  it("tries refresh on cold start when access token is missing", async () => {
    mockHasAuthToken.mockReturnValue(false);
    mockAttemptRestRecovery.mockResolvedValueOnce("recovered");
    mockFetchBootstrap.mockResolvedValue(buildBootstrap("driver:42"));
    const handle: { current: SessionHandle | null } = { current: null };
    const { renderer } = await buildHarness(handle);

    await act(async () => {
      await handle.current?.bootstrapSession();
    });

    expect(mockAttemptRestRecovery).toHaveBeenCalledWith("cold_start");
    expect(handle.current?.status).toBe("ready");
    expect(handle.current?.activeContext?.context_id).toBe("driver:42");
    await act(async () => {
      renderer.unmount();
    });
  });

  it("keeps ready status when no refresh token exists and bootstrap is unauthenticated", async () => {
    mockHasAuthToken.mockReturnValue(false);
    mockHasStoredRefreshToken.mockResolvedValue(false);
    mockFetchBootstrap.mockResolvedValue(buildUnauthenticatedBootstrap());
    const handle: { current: SessionHandle | null } = { current: null };
    const { renderer } = await buildHarness(handle);

    await act(async () => {
      await handle.current?.bootstrapSession();
    });

    expect(mockRefreshAuthTokenNow).not.toHaveBeenCalled();
    expect(handle.current?.status).toBe("ready");
    expect(handle.current?.activeContext).toBeNull();
    await act(async () => {
      renderer.unmount();
    });
  });

  it("keeps ready status when refresh fails and backend returns unauthenticated bootstrap", async () => {
    mockHasAuthToken.mockReturnValue(false);
    mockAttemptRestRecovery.mockResolvedValueOnce("no_action");
    mockFetchBootstrap.mockResolvedValue(buildUnauthenticatedBootstrap());
    const handle: { current: SessionHandle | null } = { current: null };
    const { renderer } = await buildHarness(handle);

    await act(async () => {
      await handle.current?.bootstrapSession();
    });

    expect(mockAttemptRestRecovery).toHaveBeenCalledWith("cold_start");
    expect(handle.current?.status).toBe("ready");
    expect(handle.current?.activeContext).toBeNull();
    await act(async () => {
      renderer.unmount();
    });
  });

  it("guards against concurrent bootstrap cycles", async () => {
    mockHasAuthToken.mockReturnValue(true);
    mockFetchBootstrap.mockImplementation(async () => {
      await Promise.resolve();
      return buildBootstrap("driver:42");
    });
    const handle: { current: SessionHandle | null } = { current: null };
    const { renderer } = await buildHarness(handle);

    let firstPromise: Promise<void> = Promise.resolve();
    let secondPromise: Promise<void> = Promise.resolve();
    await act(async () => {
      firstPromise = handle.current?.bootstrapSession() ?? Promise.resolve();
      secondPromise = handle.current?.bootstrapSession() ?? Promise.resolve();
    });

    await act(async () => {
      await Promise.all([firstPromise, secondPromise]);
    });

    expect(mockFetchBootstrap).toHaveBeenCalledTimes(1);
    expect(handle.current?.status).toBe("ready");
    await act(async () => {
      renderer.unmount();
    });
  });

  it("switches context and clears scoped cache for previous context", async () => {
    mockFetchBootstrap.mockResolvedValue(buildBootstrap("driver:42"));
    mockSwitchContext.mockResolvedValue({
      success: true,
      active_context_id: "client:self",
      available_contexts: buildBootstrap("client:self").available_contexts,
      feature_flags: {},
      request_id: "req-switch",
    });
    const handle: { current: SessionHandle | null } = { current: null };
    const { renderer, queryClient } = await buildHarness(handle);

    await act(async () => {
      await handle.current?.bootstrapSession();
    });
    await act(async () => {
      await handle.current?.changeContext("client:self");
    });

    expect(mockSwitchContext).toHaveBeenCalledWith("client:self");
    expect(mockApplyContextCachePolicyOnSwitch).toHaveBeenCalledWith(queryClient, "driver:42");
    expect(handle.current?.activeContext?.context_id).toBe("client:self");
    expect(mockDisconnect).toHaveBeenCalled();
    expect(mockOnContextSwitch).not.toHaveBeenCalledWith("client:self", expect.anything());
    await act(async () => {
      renderer.unmount();
    });
  });

  it("does not connect driver realtime on company bootstrap", async () => {
    mockFetchBootstrap.mockResolvedValue({
      ...buildBootstrap(null),
      available_contexts: [
        {
          context_id: "company:99",
          context_type: "company" as const,
          label: "Company",
          organization_id: 99,
          permissions: ["company:dashboard:read"],
          is_default: true,
        },
      ],
      active_context_id: "company:99",
    });
    const handle: { current: SessionHandle | null } = { current: null };
    const { renderer } = await buildHarness(handle);

    await act(async () => {
      await handle.current?.bootstrapSession();
    });

    expect(handle.current?.status).toBe("ready");
    expect(handle.current?.activeContext?.context_type).toBe("company");
    expect(mockDisconnect).toHaveBeenCalled();
    expect(mockOnContextSwitch).not.toHaveBeenCalled();
    await act(async () => {
      renderer.unmount();
    });
  });

  it("ignores driver auth exhaustion when active context is company", async () => {
    let authExhaustedCb: ((reason: "exhausted" | "terminal", code?: string) => void) | null = null;
    mockOnAuthExhausted.mockImplementation((cb) => {
      authExhaustedCb = cb;
      return () => undefined;
    });
    mockFetchBootstrap.mockResolvedValue({
      ...buildBootstrap(null),
      available_contexts: [
        {
          context_id: "company:99",
          context_type: "company" as const,
          label: "Company",
          organization_id: 99,
          permissions: ["company:dashboard:read"],
          is_default: true,
        },
      ],
      active_context_id: "company:99",
    });
    const handle: { current: SessionHandle | null } = { current: null };
    const { renderer } = await buildHarness(handle);

    await act(async () => {
      await handle.current?.bootstrapSession();
    });

    await act(async () => {
      authExhaustedCb?.("exhausted", "session_revoked");
    });

    expect(mockLogoutSession).not.toHaveBeenCalled();
    expect(handle.current?.status).toBe("ready");
    await act(async () => {
      renderer.unmount();
    });
  });

  it("keeps session coherent across login/logout and resume bootstrap", async () => {
    mockFetchBootstrap.mockImplementation(async (contextId?: string | null) => {
      if (!contextId) return buildBootstrap("driver:42");
      return buildBootstrap(contextId);
    });
    mockLogin.mockResolvedValue(undefined);
    mockLogoutSession.mockResolvedValue(undefined);

    const handle: { current: SessionHandle | null } = { current: null };
    const { renderer } = await buildHarness(handle);

    await act(async () => {
      await handle.current?.bootstrapSession();
    });
    await act(async () => {
      await handle.current?.login("driver@lirie.ch", "secret");
    });
    await act(async () => {
      handle.current?.logout();
      await handle.current?.bootstrapSession();
    });

    expect(mockLogin).toHaveBeenCalledWith("driver@lirie.ch", "secret");
    expect(mockPerformExplicitLogout).toHaveBeenCalled();
    expect(mockPurgeDriverProfileCache).toHaveBeenCalled();
    expect(mockDisconnect).toHaveBeenCalled();
    expect(mockClearAllContextCache).toHaveBeenCalled();
    expect(mockFetchBootstrap).toHaveBeenCalledWith(null);
    expect(handle.current?.status).toBe("ready");
    expect(handle.current?.activeContext?.context_id).toBe("driver:42");
    await act(async () => {
      renderer.unmount();
    });
  });

  it("rejects invalid company context invariant during bootstrap", async () => {
    mockFetchBootstrap.mockResolvedValue({
      ...buildBootstrap(null),
      available_contexts: [
        {
          context_id: "company:",
          context_type: "company" as const,
          label: "Company",
          organization_id: null,
          permissions: ["company:dashboard:read"],
          is_default: true,
        },
      ],
      active_context_id: "company:",
    });
    const handle: { current: SessionHandle | null } = { current: null };
    const { renderer } = await buildHarness(handle);

    await act(async () => {
      await handle.current?.bootstrapSession();
    });

    expect(handle.current?.status).toBe("error");
    expect(handle.current?.error).toContain("company context requires company_id");
    await act(async () => {
      renderer.unmount();
    });
  });

  it("sets error status on bootstrap transport failure", async () => {
    mockHasAuthToken.mockReturnValue(true);
    mockFetchBootstrap.mockRejectedValue(new Error("network error"));
    const handle: { current: SessionHandle | null } = { current: null };
    const { renderer } = await buildHarness(handle);

    await act(async () => {
      await handle.current?.bootstrapSession();
    });

    expect(handle.current?.status).toBe("error");
    expect(handle.current?.error).toContain("network error");
    await act(async () => {
      renderer.unmount();
    });
  });
});
