/**
 * P2.2 — Tests authLogging SRE-grade (session_id, refresh_cycle_id, dedupe, hashing).
 */

jest.mock("@react-native-async-storage/async-storage", () => ({
  setItem: jest.fn(),
  getItem: jest.fn(),
  removeItem: jest.fn(),
}));

jest.mock("expo-crypto", () => ({
  randomUUID: () => "mock-uuid-" + Math.random().toString(36).slice(2, 10),
  digestStringAsync: jest.fn().mockResolvedValue("a1b2c3d4e5f67890"), // 12+ chars pour hash
  CryptoDigestAlgorithm: { SHA256: "SHA-256" },
}));

const mockGetLogContextSnapshot = jest.fn(() => ({
  platform: "ios",
  build_number: "1",
  app_version: "1.0.0",
}));
const mockGetNetworkStateSnapshot = jest.fn(() => null);
let mockUseRealLogContextForSnapshot = false;

jest.mock("../logContext", () => {
  const actual = jest.requireActual<typeof import("../logContext")>("../logContext");
  return {
    setLogContextUser: actual.setLogContextUser,
    getLogContextSnapshot: () =>
      mockUseRealLogContextForSnapshot ? actual.getLogContextSnapshot() : mockGetLogContextSnapshot(),
    initLogContext: actual.initLogContext,
  };
});
jest.mock("../networkState", () => ({
  getNetworkStateSnapshot: () => mockGetNetworkStateSnapshot(),
}));

import { logAuthEvent, beginRefreshCycle, getCurrentRefreshCycleId } from "../authLogging";
import { setLogContextUser, getLogContextSnapshot } from "../logContext";

describe("authLogging P2.2 — session_id", () => {
  let logSpy: jest.SpyInstance;
  let debugSpy: jest.SpyInstance;

  beforeEach(() => {
    jest.clearAllMocks();
    logSpy = jest.spyOn(console, "log").mockImplementation(() => {});
    debugSpy = jest.spyOn(console, "debug").mockImplementation(() => {});
  });

  afterEach(() => {
    logSpy.mockRestore();
    debugSpy.mockRestore();
  });

  const getLastLoggedJson = (): Record<string, unknown> => {
    const logCalls = logSpy.mock.calls.concat(debugSpy.mock.calls);
    const authLogCall = logCalls.find((c) => c[0] === "[AUTH_LOG]");
    if (!authLogCall || !authLogCall[1]) return {};
    try {
      return JSON.parse(authLogCall[1] as string);
    } catch {
      return {};
    }
  };

  it("session_id est présent sur tous les logs", () => {
    logAuthEvent("TEST_EVENT", { foo: "bar" });
    const logged = getLastLoggedJson();
    expect(logged.session_id).toBeDefined();
    expect(typeof logged.session_id).toBe("string");
    expect((logged.session_id as string).length).toBeGreaterThan(0);
  });

  it("session_id est stable entre deux appels", () => {
    logAuthEvent("EVENT_1", {});
    const logged1 = getLastLoggedJson();
    logAuthEvent("EVENT_2", { route: "x" }); // payload différent pour éviter dedupe
    const logged2 = getLastLoggedJson();
    expect(logged1.session_id).toBe(logged2.session_id);
  });
});

describe("authLogging P2.2 — refresh_cycle_id", () => {
  let logSpy: jest.SpyInstance;
  let debugSpy: jest.SpyInstance;

  beforeEach(() => {
    jest.clearAllMocks();
    logSpy = jest.spyOn(console, "log").mockImplementation(() => {});
    debugSpy = jest.spyOn(console, "debug").mockImplementation(() => {});
  });

  afterEach(() => {
    logSpy.mockRestore();
    debugSpy.mockRestore();
  });

  const getAllLoggedJson = (): Record<string, unknown>[] => {
    const logCalls = [...logSpy.mock.calls, ...debugSpy.mock.calls];
    return logCalls
      .filter((c) => c[0] === "[AUTH_LOG]")
      .map((c) => {
        try {
          return JSON.parse((c[1] as string) || "{}");
        } catch {
          return {};
        }
      });
  };

  it("refresh_cycle_id: start/success/fail portent le même id", () => {
    const cycleId = beginRefreshCycle("driver");
    expect(cycleId).toBe(getCurrentRefreshCycleId());

    logAuthEvent("AUTH_REFRESH_START", { route: "driver", trigger: "api_401" });
    logAuthEvent("AUTH_REFRESH_SUCCESS", { route: "driver" });
    logAuthEvent("AUTH_REFRESH_FAIL", { route: "driver", status: 401, outcome: "logout" });

    const all = getAllLoggedJson();
    const withCycleId = all.filter((o) => "refresh_cycle_id" in o);
    expect(withCycleId.length).toBe(3);
    const ids = withCycleId.map((o) => o.refresh_cycle_id);
    expect(ids[0]).toBe(ids[1]);
    expect(ids[1]).toBe(ids[2]);
    expect(ids[0]).toBe(cycleId);
  });
});

describe("authLogging P2.2 — hashing (logContext)", () => {
  beforeEach(() => {
    mockUseRealLogContextForSnapshot = true;
  });
  afterEach(() => {
    mockUseRealLogContextForSnapshot = false;
  });

  it("user_public_id_hash ne contient pas la valeur brute", async () => {
    const rawId = "user-public-id-secret-12345";
    setLogContextUser({ user_public_id: rawId });
    await new Promise((r) => setTimeout(r, 100)); // attendre hash async
    const ctx = getLogContextSnapshot();
    expect(ctx.user_public_id_hash).toBeDefined();
    expect(ctx.user_public_id_hash).not.toBe(rawId);
    expect((ctx.user_public_id_hash as string).length).toBeLessThanOrEqual(12);
  });

  it("device_id_hash ne contient pas la valeur brute", async () => {
    const rawDeviceId = "device-abc-secret-xyz";
    setLogContextUser({ device_id: rawDeviceId });
    await new Promise((r) => setTimeout(r, 100));
    const ctx = getLogContextSnapshot();
    expect(ctx.device_id_hash).toBeDefined();
    expect(ctx.device_id_hash).not.toBe(rawDeviceId);
    expect((ctx.device_id_hash as string).length).toBeLessThanOrEqual(12);
  });
});

describe("authLogging P2.2 — dedupe", () => {
  let logSpy: jest.SpyInstance;
  let debugSpy: jest.SpyInstance;

  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers();
    logSpy = jest.spyOn(console, "log").mockImplementation(() => {});
    debugSpy = jest.spyOn(console, "debug").mockImplementation(() => {});
  });

  afterEach(() => {
    jest.useRealTimers();
    logSpy.mockRestore();
    debugSpy.mockRestore();
  });

  const countAuthLogCalls = (): number => {
    const logCalls = logSpy.mock.calls.concat(debugSpy.mock.calls);
    return logCalls.filter((c) => c[0] === "[AUTH_LOG]").length;
  };

  it("2 logs identiques <5s => 1 seul émis", () => {
    const payload = { route: "r", outcome: "o", status: "s" };
    logAuthEvent("EVENT_DEDUPE", payload);
    expect(countAuthLogCalls()).toBe(1);
    logAuthEvent("EVENT_DEDUPE", payload);
    expect(countAuthLogCalls()).toBe(1);
  });

  it("2 logs identiques >5s => 2 émis", () => {
    const payload = { route: "r", outcome: "o", status: "s" };
    logAuthEvent("EVENT_DEDUPE_2", payload);
    expect(countAuthLogCalls()).toBe(1);
    jest.advanceTimersByTime(6000);
    logAuthEvent("EVENT_DEDUPE_2", payload);
    expect(countAuthLogCalls()).toBe(2);
  });

  it("LOGOUT_TRANSITION n'est jamais dedupé", () => {
    const payload = { route: "driver", reason: "manual" };
    logAuthEvent("LOGOUT_TRANSITION", payload);
    logAuthEvent("LOGOUT_TRANSITION", payload);
    expect(countAuthLogCalls()).toBe(2);
  });

  it("role distinct : driver vs enterprise ne se dédupliquent pas", () => {
    const basePayload = { route: "", outcome: "error", status: 401 };
    logAuthEvent("SOCKET_CONNECT_ERROR", { ...basePayload, role: "driver" });
    logAuthEvent("SOCKET_CONNECT_ERROR", { ...basePayload, role: "enterprise" });
    expect(countAuthLogCalls()).toBe(2);
  });
});

describe("authLogging P2.2 — sanitization (pas de secrets)", () => {
  let logSpy: jest.SpyInstance;
  let debugSpy: jest.SpyInstance;

  beforeEach(() => {
    jest.clearAllMocks();
    logSpy = jest.spyOn(console, "log").mockImplementation(() => {});
    debugSpy = jest.spyOn(console, "debug").mockImplementation(() => {});
  });

  afterEach(() => {
    logSpy.mockRestore();
    debugSpy.mockRestore();
  });

  const getLastLoggedJson = (): Record<string, unknown> => {
    const logCalls = [...logSpy.mock.calls, ...debugSpy.mock.calls];
    const authLogCalls = logCalls.filter((c) => c[0] === "[AUTH_LOG]");
    const lastCall = authLogCalls[authLogCalls.length - 1];
    if (!lastCall || !lastCall[1]) return {};
    try {
      return JSON.parse(lastCall[1] as string);
    } catch {
      return {};
    }
  };

  it("token n'apparaît pas dans le log", () => {
    logAuthEvent("TEST", { token: "secret-jwt-xyz", route: "driver" });
    const logged = getLastLoggedJson();
    expect(logged.token).toBeUndefined();
    expect(JSON.stringify(logged)).not.toContain("secret-jwt");
  });

  it("password n'apparaît pas dans le log", () => {
    logAuthEvent("TEST", { password: "secret123", route: "driver" });
    const logged = getLastLoggedJson();
    expect(logged.password).toBeUndefined();
  });
});
