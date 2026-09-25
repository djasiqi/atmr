/**
 * Pont warm recovery — single entry pour 401 / socket / foreground.
 */
import { beforeEach, describe, expect, it, jest } from "@jest/globals";

const mockAttemptRestRecovery = jest.fn();

jest.mock("./authRecoveryCoordinator", () => ({
  attemptRestRecovery: (...args: unknown[]) => mockAttemptRestRecovery(...args),
}));

import {
  requestWarmAuthRecovery,
  setWarmAuthRecoveryHandler,
} from "./authWarmRecoveryBridge";

describe("authWarmRecoveryBridge", () => {
  beforeEach(() => {
    setWarmAuthRecoveryHandler(null);
    mockAttemptRestRecovery.mockReset();
    mockAttemptRestRecovery.mockResolvedValue("recovered");
  });

  it("sans handler → attemptRestRecovery", async () => {
    const outcome = await requestWarmAuthRecovery("api_401");
    expect(outcome).toBe("recovered");
    expect(mockAttemptRestRecovery).toHaveBeenCalledWith("api_401");
  });

  it("avec handler → délègue (SessionProvider)", async () => {
    const handler = jest.fn(async () => "keep_local" as const);
    setWarmAuthRecoveryHandler(handler);
    const outcome = await requestWarmAuthRecovery("socket_auth_failure");
    expect(outcome).toBe("keep_local");
    expect(handler).toHaveBeenCalledWith("socket_auth_failure");
    expect(mockAttemptRestRecovery).not.toHaveBeenCalled();
  });
});
