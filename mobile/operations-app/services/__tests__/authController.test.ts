import {
  invokeForceLogoutDriver,
  registerForceLogoutDriver,
} from "../authController";

jest.mock("../authLogging", () => ({
  logAuthEvent: jest.fn(),
}));

describe("authController force logout contract", () => {
  it("rejette un forceLogout sans metadata obligatoire", async () => {
    await expect(
      invokeForceLogoutDriver({
        reason: "refresh_invalid",
        // @ts-expect-error intentionally missing metadata fields
      })
    ).rejects.toThrow("metadata incomplete");
  });

  it("propage la metadata complète jusqu'au callback final", async () => {
    const cb = jest.fn();
    const unreg = registerForceLogoutDriver(cb);
    await invokeForceLogoutDriver({
      reason: "refresh_invalid",
      severity: "AUTH_HARD_FAILURE",
      source: "driver",
      trigger_source: "api_interceptor",
      tenant_id: 123,
      session_id: "sess-1",
    });
    expect(cb).toHaveBeenCalledWith(
      "refresh_invalid",
      expect.objectContaining({
        reason: "refresh_invalid",
        severity: "AUTH_HARD_FAILURE",
        source: "driver",
        trigger_source: "api_interceptor",
        tenant_id: 123,
        session_id: "sess-1",
      })
    );
    unreg();
  });
});
