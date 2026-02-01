/**
 * P1.B — Tests AUTH_INVALID unifié
 * Vérifie que seul 401/403 (auth invalide) déclenche logout, pas réseau/5xx.
 * Tests des helpers + flow refreshProfile via mocks.
 */

import { isNetworkError, isHttpAuthError, getHttpStatus } from "@/utils/authErrorHelpers";

// Mock minimal pour tester la logique de classification
describe("P1.B — authErrorHelpers", () => {
  describe("isHttpAuthError", () => {
    it("retourne true pour 401", () => {
      expect(isHttpAuthError({ response: { status: 401 } })).toBe(true);
    });
    it("retourne true pour 403", () => {
      expect(isHttpAuthError({ response: { status: 403 } })).toBe(true);
    });
    it("retourne false pour 500", () => {
      expect(isHttpAuthError({ response: { status: 500 } })).toBe(false);
    });
    it("retourne false pour erreur réseau (pas de response)", () => {
      expect(isHttpAuthError({ code: "ERR_NETWORK" })).toBe(false);
    });
  });

  describe("isNetworkError", () => {
    it("retourne true pour pas de response", () => {
      expect(isNetworkError({ code: "ERR_NETWORK" })).toBe(true);
    });
    it("retourne true pour ECONNABORTED (timeout)", () => {
      expect(isNetworkError({ code: "ECONNABORTED" })).toBe(true);
    });
    it("retourne false pour 401", () => {
      expect(isNetworkError({ response: { status: 401 } })).toBe(false);
    });
  });

  describe("getHttpStatus", () => {
    it("extrait 401", () => {
      expect(getHttpStatus({ response: { status: 401 } })).toBe(401);
    });
    it("retourne null pour erreur réseau", () => {
      expect(getHttpStatus({ code: "ERR_NETWORK" })).toBe(null);
    });
  });
});

// Tests du flow refreshProfile (logique simulée)
describe("P1.B — refreshProfile flow (logique)", () => {
  it("401 + retry success => PAS de logout (simulation)", async () => {
    const mockProfile = { id: 1, first_name: "Test", last_name: "Driver" };
    // Simule: initial fetch a échoué 401, on est dans le catch. Retry (refresh+fetch) réussit.
    const mockFetchRetry = async () => mockProfile;
    const mockRefresh = async () => "new-token";

    const error = { response: { status: 401 } };
    let didLogout = false;
    if (isHttpAuthError(error)) {
      try {
        await mockRefresh();
        const profile = await mockFetchRetry();
        expect(profile).toEqual(mockProfile);
      } catch (retryError) {
        if (isHttpAuthError(retryError)) {
          didLogout = true;
        }
      }
    }
    expect(didLogout).toBe(false);
  });

  it("401 + retry 401 => logout (simulation)", async () => {
    const mockFetch = async () => {
      throw { response: { status: 401 } };
    };
    const mockRefresh = async () => "new-token";

    const error = { response: { status: 401 } };
    let didLogout = false;
    if (isHttpAuthError(error)) {
      try {
        await mockRefresh();
        await mockFetch();
      } catch (retryError) {
        if (isHttpAuthError(retryError)) {
          didLogout = true;
        }
      }
    }
    expect(didLogout).toBe(true);
  });

  it("erreur réseau => isHttpAuthError false, isNetworkError true (pas de logout)", () => {
    const error = { code: "ERR_NETWORK", message: "Network Error" };
    expect(isHttpAuthError(error)).toBe(false);
    expect(isNetworkError(error)).toBe(true);
  });
});
