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
  buildAuthNamespace,
  sanitizeSegment,
} from "./keys";

describe("storage/keys — P1.A garde-fou", () => {
  describe("DRIVER_AUTH_KEYS", () => {
    it("doit contenir les clés SecureStore attendues (access, refresh, user_public_id)", () => {
      expect(DRIVER_AUTH_SECURE_KEYS).toContain("driver_refresh_token");
      expect(DRIVER_AUTH_SECURE_KEYS).toContain("driver_access_token");
      expect(DRIVER_AUTH_SECURE_KEYS).toContain("driver_user_public_id");
      expect(DRIVER_AUTH_SECURE_KEYS).toContain("driver_refresh_token_backup");
      expect(DRIVER_AUTH_SECURE_KEYS).toHaveLength(4);
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

  describe("sanitizeSegment", () => {
    it("laisse passer les caractères autorisés SecureStore", () => {
      expect(sanitizeSegment("abc123")).toBe("abc123");
      expect(sanitizeSegment("a.b-c_d")).toBe("a.b-c_d");
    });

    it("remplace les caractères interdits par _", () => {
      expect(sanitizeSegment("a:b")).toBe("a_b");
      expect(sanitizeSegment("a/b@c!d")).toBe("a_b_c_d");
    });

    it("retourne _ pour une chaîne vide ou falsy", () => {
      expect(sanitizeSegment("")).toBe("_");
    });
  });

  describe("buildAuthNamespace", () => {
    it("utilise . comme séparateur (pas :)", () => {
      const ns = buildAuthNamespace({ role: "driver", userId: "123", tenantId: 456, sessionId: "s1" });
      expect(ns).toBe("driver.123.456.s1");
      expect(ns).not.toContain(":");
    });

    it("ne contient que des caractères valides SecureStore", () => {
      const ns = buildAuthNamespace({ role: "enterprise", userId: "u@b", tenantId: "t:1", sessionId: null });
      expect(ns).toMatch(/^[A-Za-z0-9._-]+$/);
      expect(ns).not.toContain(":");
      expect(ns).not.toContain("@");
    });

    it("utilise none pour tenant/session absents", () => {
      const ns = buildAuthNamespace({ role: "driver", userId: "42" });
      expect(ns).toBe("driver.42.none.none");
    });

    it("gère userId undefined/null avec fallback", () => {
      const ns = buildAuthNamespace({ role: "driver", userId: undefined as any });
      expect(ns).toMatch(/^driver\..+\.none\.none$/);
      expect(ns).not.toContain("undefined");
    });

    it("produit des clés compatibles SecureStore quand concaténé", () => {
      const ns = buildAuthNamespace({ role: "driver", userId: "123", tenantId: 456, sessionId: "s1" });
      const fullKey = `driver_access_token.ns.${ns}`;
      expect(fullKey).toMatch(/^[A-Za-z0-9._-]+$/);
    });
  });
});
