/**
 * Tests pour authSync : invariant isAuthReady.
 * Bootstrap sans session = on ne appelle jamais notifyAuthReady → isAuthReady reste false.
 */
import {
  notifyAuthReady,
  notifyAuthNotReady,
  isAuthReadySync,
  waitForAuthReady,
} from "../authSync";

describe("authSync", () => {
  beforeEach(() => {
    // Réinitialiser l'état pour chaque test (notifyAuthNotReady vide la queue et met isAuthReady à false)
    notifyAuthNotReady();
  });

  describe("invariant isAuthReady", () => {
    it("isAuthReady reste false tant qu'on n'a pas appelé notifyAuthReady (bootstrap sans session)", () => {
      expect(isAuthReadySync()).toBe(false);
      // Simuler qu'aucune session n'a été restaurée → on n'appelle jamais notifyAuthReady
      expect(isAuthReadySync()).toBe(false);
    });

    it("isAuthReady passe à true après notifyAuthReady()", () => {
      expect(isAuthReadySync()).toBe(false);
      notifyAuthReady();
      expect(isAuthReadySync()).toBe(true);
    });

    it("après notifyAuthNotReady(), isAuthReady repasse à false (ex: après clearAll / refresh 401)", () => {
      notifyAuthReady();
      expect(isAuthReadySync()).toBe(true);
      notifyAuthNotReady();
      expect(isAuthReadySync()).toBe(false);
    });
  });

  describe("waitForAuthReady", () => {
    it("résout immédiatement si isAuthReady est déjà true", async () => {
      notifyAuthReady();
      await expect(waitForAuthReady(100)).resolves.toBeUndefined();
    });

    it("rejette après timeout si isAuthReady reste false", async () => {
      await expect(waitForAuthReady(50)).rejects.toThrow(/Timeout/);
    });
  });
});
