/**
 * P1.C — Tests logoutMarker (setLogoutMarker, consumeLogoutMarker, TTL).
 */

import AsyncStorage from "@react-native-async-storage/async-storage";
import {
  setLogoutMarker,
  consumeLogoutMarker,
  isSessionExpiredReason,
} from "../logoutMarker";

jest.mock("@react-native-async-storage/async-storage", () => ({
  setItem: jest.fn(),
  getItem: jest.fn(),
  removeItem: jest.fn(),
}));

describe("logoutMarker", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    (AsyncStorage.getItem as jest.Mock).mockResolvedValue(null);
    (AsyncStorage.setItem as jest.Mock).mockResolvedValue(undefined);
    (AsyncStorage.removeItem as jest.Mock).mockResolvedValue(undefined);
  });

  describe("isSessionExpiredReason", () => {
    it("retourne true pour refresh_rejected_401", () => {
      expect(isSessionExpiredReason("refresh_rejected_401")).toBe(true);
    });
    it("retourne true pour refresh_rejected_403", () => {
      expect(isSessionExpiredReason("refresh_rejected_403")).toBe(true);
    });
    it("retourne true pour profile_auth_invalid", () => {
      expect(isSessionExpiredReason("profile_auth_invalid")).toBe(true);
    });
    it("retourne false pour manual_logout", () => {
      expect(isSessionExpiredReason("manual_logout")).toBe(false);
    });
  });

  describe("setLogoutMarker + consumeLogoutMarker", () => {
    it("setLogoutMarker puis consumeLogoutMarker retourne le marker puis supprime", async () => {
      await setLogoutMarker({
        route: "driver",
        reason: "profile_auth_invalid",
        ts: Date.now(),
      });

      expect(AsyncStorage.setItem).toHaveBeenCalledWith(
        "auth.logout_marker",
        expect.stringContaining("profile_auth_invalid")
      );

      (AsyncStorage.getItem as jest.Mock).mockResolvedValue(
        JSON.stringify({
          route: "driver",
          reason: "profile_auth_invalid",
          ts: Date.now(),
        })
      );

      const marker = await consumeLogoutMarker("driver");
      expect(marker).not.toBeNull();
      expect(marker?.route).toBe("driver");
      expect(marker?.reason).toBe("profile_auth_invalid");
      expect(AsyncStorage.removeItem).toHaveBeenCalledWith("auth.logout_marker");
    });

    it("consumeLogoutMarker retourne null si route ne correspond pas", async () => {
      (AsyncStorage.getItem as jest.Mock).mockResolvedValue(
        JSON.stringify({
          route: "enterprise",
          reason: "refresh_rejected_403",
          ts: Date.now(),
        })
      );

      const marker = await consumeLogoutMarker("driver");
      expect(marker).toBeNull();
    });

    it("consumeLogoutMarker retourne null si marker expiré (TTL > 5 min)", async () => {
      const oldTs = Date.now() - 6 * 60 * 1000;
      (AsyncStorage.getItem as jest.Mock).mockResolvedValue(
        JSON.stringify({
          route: "driver",
          reason: "profile_auth_invalid",
          ts: oldTs,
        })
      );

      const marker = await consumeLogoutMarker("driver");
      expect(marker).toBeNull();
      expect(AsyncStorage.removeItem).toHaveBeenCalledWith("auth.logout_marker");
    });
  });
});
