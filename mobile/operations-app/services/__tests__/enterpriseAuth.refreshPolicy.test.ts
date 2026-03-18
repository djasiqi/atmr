import {
  getAuthFailureReason,
  shouldLogoutFromRefreshFailure,
} from "../authGuards";

describe("enterprise refresh policy", () => {
  it("logout sur hard failure refresh_invalid", () => {
    const error = {
      response: {
        status: 401,
        data: { reason: "refresh_invalid" },
      },
    };
    const decision = shouldLogoutFromRefreshFailure(error, "refresh_endpoint");
    expect(getAuthFailureReason(error)).toBe("refresh_invalid");
    expect(decision.shouldLogout).toBe(true);
    expect(decision.severity).toBe("AUTH_HARD_FAILURE");
  });

  it("ne logout pas sur erreur transitoire 503", () => {
    const error = {
      response: {
        status: 503,
        data: { error: "service_unavailable" },
      },
    };
    const decision = shouldLogoutFromRefreshFailure(error, "refresh_endpoint");
    expect(getAuthFailureReason(error)).toBe("server_error");
    expect(decision.shouldLogout).toBe(false);
    expect(decision.severity).toBe("AUTH_SOFT_FAILURE");
  });

  it("401 sans reason structurée => unknown_refresh_401 soft", () => {
    const error = { response: { status: 401, data: {} } };
    const decision = shouldLogoutFromRefreshFailure(error, "refresh_endpoint");
    expect(getAuthFailureReason(error)).toBe("unknown_refresh_401");
    expect(decision.shouldLogout).toBe(false);
  });
});
