import { describe, expect, it } from "@jest/globals";
import { AUTH_SECURE_STORE_KEYS } from "./authCredentialStore";

/** Même règle que expo-secure-store `isValidKey`. */
const SECURE_STORE_KEY_RE = /^[\w.-]+$/;

describe("authCredentialStore SecureStore keys", () => {
  it("n'utilise que des clés compatibles expo-secure-store", () => {
    expect(AUTH_SECURE_STORE_KEYS.length).toBeGreaterThan(0);
    for (const key of AUTH_SECURE_STORE_KEYS) {
      expect(key).toMatch(SECURE_STORE_KEY_RE);
      expect(key).not.toMatch(/[@/:]/);
    }
  });

  it("rejette le format legacy @atmr/auth/...", () => {
    expect(SECURE_STORE_KEY_RE.test("@atmr/auth/installation_id")).toBe(false);
    expect(SECURE_STORE_KEY_RE.test("atmr.auth.installation_id")).toBe(true);
  });
});
