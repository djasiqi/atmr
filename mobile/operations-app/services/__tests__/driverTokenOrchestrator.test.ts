import { refreshDriverTokenOrchestrated } from "@/services/driverTokenOrchestrator";

jest.mock("@react-native-async-storage/async-storage", () => ({
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn().mockResolvedValue(undefined),
  multiRemove: jest.fn().mockResolvedValue(undefined),
}));

jest.mock("expo-secure-store", () => ({
  setItemAsync: jest.fn().mockResolvedValue(undefined),
  getItemAsync: jest.fn().mockResolvedValue(null),
  deleteItemAsync: jest.fn().mockResolvedValue(undefined),
}));

jest.mock("@/services/api", () => ({
  runDriverRefreshSingleflight: jest.fn(),
}));

jest.mock("@/services/authLogging", () => ({
  beginRefreshCycle: jest.fn(() => "cycle-test-1"),
  logAuthEvent: jest.fn(),
}));

describe("driverTokenOrchestrator", () => {
  const { runDriverRefreshSingleflight } = jest.requireMock("@/services/api") as {
    runDriverRefreshSingleflight: jest.Mock;
  };
  const { logAuthEvent } = jest.requireMock("@/services/authLogging") as {
    logAuthEvent: jest.Mock;
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("doit logguer start/success et retourner le token", async () => {
    runDriverRefreshSingleflight.mockResolvedValue("token-new");

    const token = await refreshDriverTokenOrchestrated("socket_connect_error");

    expect(token).toBe("token-new");
    expect(runDriverRefreshSingleflight).toHaveBeenCalledTimes(1);
    expect(logAuthEvent).toHaveBeenCalledWith(
      "AUTH_REFRESH_START",
      expect.objectContaining({
        route: "driver",
        trigger_source: "socket_connect_error",
      })
    );
    expect(logAuthEvent).toHaveBeenCalledWith(
      "AUTH_REFRESH_SUCCESS",
      expect.objectContaining({
        route: "driver",
        trigger_source: "socket_connect_error",
        outcome: "token_refreshed",
      })
    );
  });

  it("doit logguer fail et propager l'erreur", async () => {
    const error = Object.assign(new Error("refresh failed"), {
      response: { status: 401 },
    });
    runDriverRefreshSingleflight.mockRejectedValue(error);

    await expect(
      refreshDriverTokenOrchestrated("socket_unauthorized")
    ).rejects.toThrow("refresh failed");

    expect(logAuthEvent).toHaveBeenCalledWith(
      "AUTH_REFRESH_FAIL",
      expect.objectContaining({
        route: "driver",
        trigger_source: "socket_unauthorized",
        status: 401,
      })
    );
  });
});
