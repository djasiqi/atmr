/**
 * Tests pour authGuards : messages UX et dedupe AUTH_NOT_READY.
 * Couvre : bootstrap sans session (message propre), dedupe popup, raisons.
 */
import {
  AuthNotReadyError,
  getAuthFailureReason,
  isAuthNotReadyError,
  isHardAuthFailure,
  shouldLogoutFromRefreshFailure,
  getAuthNotReadyDisplayMessage,
  shouldShowAuthNotReadyAlert,
} from "../authGuards";

describe("authGuards", () => {
  describe("getAuthNotReadyDisplayMessage", () => {
    it("retourne un message UX propre pour missing_access_token (pas le message technique)", () => {
      const err = new AuthNotReadyError({
        kind: "driver",
        reason: "missing_access_token",
        url: "/driver/me/bookings/30300/status",
      });
      const msg = getAuthNotReadyDisplayMessage(err);
      expect(msg).toBe("Session non prête. Veuillez patienter ou vous reconnecter.");
      expect(msg).not.toContain("AUTH_NOT_READY");
      expect(msg).not.toContain("missing_access_token");
    });

    it("retourne un message pour missing_refresh_token", () => {
      const err = new AuthNotReadyError({
        kind: "enterprise",
        reason: "missing_refresh_token",
      });
      expect(getAuthNotReadyDisplayMessage(err)).toBe(
        "Session expirée ou inexistante. Veuillez vous reconnecter."
      );
    });

    it("retourne null pour une erreur non AUTH_NOT_READY", () => {
      expect(getAuthNotReadyDisplayMessage(new Error("Network"))).toBeNull();
    });
  });

  describe("shouldShowAuthNotReadyAlert", () => {
    it("retourne true pour une erreur AUTH_NOT_READY sans silentDedupe (premier popup)", () => {
      const err = new AuthNotReadyError({
        kind: "driver",
        reason: "missing_access_token",
        url: "/driver/me/bookings/30300/status",
      });
      expect(shouldShowAuthNotReadyAlert(err)).toBe(true);
    });

    it("retourne false pour une erreur AUTH_NOT_READY avec silentDedupe (dedupe anti-spam)", () => {
      const err = new AuthNotReadyError({
        kind: "driver",
        reason: "missing_access_token",
        url: "/driver/me/bookings/30300/status",
        silentDedupe: true,
      });
      expect(shouldShowAuthNotReadyAlert(err)).toBe(false);
    });

    it("retourne true pour une erreur non AUTH_NOT_READY", () => {
      expect(shouldShowAuthNotReadyAlert(new Error("Other"))).toBe(true);
    });
  });

  describe("isAuthNotReadyError", () => {
    it("reconnaît une AuthNotReadyError", () => {
      const err = new AuthNotReadyError({
        kind: "driver",
        reason: "missing_access_token",
      });
      expect(isAuthNotReadyError(err)).toBe(true);
    });

    it("rejette une erreur classique", () => {
      expect(isAuthNotReadyError(new Error("Network"))).toBe(false);
    });
  });

  describe("classification hard/soft", () => {
    it("classifie refresh_invalid en hard failure", () => {
      const err = {
        response: {
          status: 401,
          data: { reason: "refresh_invalid" },
        },
      };
      expect(getAuthFailureReason(err)).toBe("refresh_invalid");
      expect(isHardAuthFailure(err, "refresh_endpoint")).toBe(true);
      expect(shouldLogoutFromRefreshFailure(err, "refresh_endpoint").shouldLogout).toBe(true);
    });

    it("classifie 401 sans reason en unknown_refresh_401 (soft)", () => {
      const err = { response: { status: 401, data: {} } };
      const decision = shouldLogoutFromRefreshFailure(err, "refresh_endpoint");
      expect(getAuthFailureReason(err)).toBe("unknown_refresh_401");
      expect(decision.shouldLogout).toBe(false);
    });
  });
});
