import { getSocketAuthFailureDecision } from "../socketAuthUtils";

describe("socket auth failure decision", () => {
  it("classe refresh_invalid en hard failure", () => {
    const decision = getSocketAuthFailureDecision({
      data: { status: 401, reason: "refresh_invalid" },
      message: "Unauthorized",
    });
    expect(decision.shouldLogout).toBe(true);
    expect(decision.reason).toBe("refresh_invalid");
    expect(decision.severity).toBe("AUTH_HARD_FAILURE");
  });

  it("classe les erreurs transport en soft failure", () => {
    const decision = getSocketAuthFailureDecision({
      message: "transport error: websocket closed",
    });
    expect(decision.shouldLogout).toBe(false);
    expect(decision.severity).toBe("AUTH_SOFT_FAILURE");
  });
});
