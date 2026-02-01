/**
 * P1.A — Garde-fou : vérifie que les listes de clés auth sont exhaustives.
 * Évite qu'une nouvelle clé auth soit oubliée lors d'un logout.
 */

import {
  DRIVER_AUTH_KEYS,
  DRIVER_AUTH_SECURE_KEYS,
  DRIVER_AUTH_ASYNC_KEYS,
  ENTERPRISE_AUTH_KEYS,
  ENTERPRISE_AUTH_SECURE_KEYS,
  ENTERPRISE_AUTH_ASYNC_KEYS,
} from "./keys";

describe("storage/keys — P1.A garde-fou", () => {
  describe("DRIVER_AUTH_KEYS", () => {
    it("doit contenir les clés SecureStore attendues (access, refresh, user_public_id)", () => {
      expect(DRIVER_AUTH_SECURE_KEYS).toContain("driver_refresh_token");
      expect(DRIVER_AUTH_SECURE_KEYS).toContain("driver_access_token");
      expect(DRIVER_AUTH_SECURE_KEYS).toContain("driver_user_public_id");
      expect(DRIVER_AUTH_SECURE_KEYS).toHaveLength(3);
    });

    it("doit contenir les clés AsyncStorage attendues (driver_id, driver_account_info)", () => {
      expect(DRIVER_AUTH_ASYNC_KEYS).toContain("driver_id");
      expect(DRIVER_AUTH_ASYNC_KEYS).toContain("enterprise.driver_account_info");
      expect(DRIVER_AUTH_ASYNC_KEYS).toHaveLength(2);
    });

    it("doit avoir une structure cohérente", () => {
      expect(DRIVER_AUTH_KEYS.secure).toEqual(DRIVER_AUTH_SECURE_KEYS);
      expect(DRIVER_AUTH_KEYS.async).toEqual(DRIVER_AUTH_ASYNC_KEYS);
    });
  });

  describe("ENTERPRISE_AUTH_KEYS", () => {
    it("doit contenir les clés SecureStore attendues (enterprise_token, enterprise_refresh)", () => {
      expect(ENTERPRISE_AUTH_SECURE_KEYS).toContain("enterprise.token");
      expect(ENTERPRISE_AUTH_SECURE_KEYS).toContain("enterprise.refresh");
      expect(ENTERPRISE_AUTH_SECURE_KEYS).toHaveLength(2);
    });

    it("doit contenir les clés AsyncStorage attendues (session, enterprise_session_just_created)", () => {
      expect(ENTERPRISE_AUTH_ASYNC_KEYS).toContain("enterprise.session");
      expect(ENTERPRISE_AUTH_ASYNC_KEYS).toContain(
        "enterprise_session_just_created"
      );
      expect(ENTERPRISE_AUTH_ASYNC_KEYS).toHaveLength(2);
    });

    it("doit avoir une structure cohérente", () => {
      expect(ENTERPRISE_AUTH_KEYS.secure).toEqual(ENTERPRISE_AUTH_SECURE_KEYS);
      expect(ENTERPRISE_AUTH_KEYS.async).toEqual(ENTERPRISE_AUTH_ASYNC_KEYS);
    });
  });

  describe("exhaustivité", () => {
    it("aucune clé ne doit être vide", () => {
      const allKeys = [
        ...DRIVER_AUTH_SECURE_KEYS,
        ...DRIVER_AUTH_ASYNC_KEYS,
        ...ENTERPRISE_AUTH_SECURE_KEYS,
        ...ENTERPRISE_AUTH_ASYNC_KEYS,
      ];
      allKeys.forEach((k) => {
        expect(k).toBeTruthy();
        expect(typeof k).toBe("string");
        expect(k.length).toBeGreaterThan(0);
      });
    });

    it("aucune clé ne doit être dupliquée entre driver et enterprise", () => {
      const driverKeys = new Set<string>([
        ...DRIVER_AUTH_SECURE_KEYS,
        ...DRIVER_AUTH_ASYNC_KEYS,
      ]);
      const enterpriseKeys = [
        ...ENTERPRISE_AUTH_SECURE_KEYS,
        ...ENTERPRISE_AUTH_ASYNC_KEYS,
      ];
      enterpriseKeys.forEach((k) => {
        expect(driverKeys.has(k)).toBe(false);
      });
    });
  });
});
