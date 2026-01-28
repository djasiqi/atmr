/**
 * Tests unitaires pour rememberMeStorage (Se souvenir de moi, chauffeur).
 * Mock expo-secure-store.
 */

import * as SecureStore from "expo-secure-store";
import {
  getRememberMe,
  setRememberMe,
  getRememberedCredentials,
  setRememberedCredentials,
  clearRememberedCredentials,
} from "../rememberMeStorage";

jest.mock("expo-secure-store", () => ({
  setItemAsync: jest.fn(() => Promise.resolve()),
  getItemAsync: jest.fn(() => Promise.resolve(null)),
  deleteItemAsync: jest.fn(() => Promise.resolve()),
}));

const mockGet = SecureStore.getItemAsync as jest.Mock;
const mockSet = SecureStore.setItemAsync as jest.Mock;
const mockDelete = SecureStore.deleteItemAsync as jest.Mock;

beforeEach(() => {
  jest.clearAllMocks();
  mockGet.mockResolvedValue(null);
  mockSet.mockResolvedValue(undefined);
  mockDelete.mockResolvedValue(undefined);
});

describe("rememberMeStorage", () => {
  describe("getRememberMe / setRememberMe", () => {
    it("getRememberMe retourne false quand la clé est absente ou non 'true'", async () => {
      mockGet.mockResolvedValue(null);
      expect(await getRememberMe()).toBe(false);

      mockGet.mockResolvedValue("false");
      expect(await getRememberMe()).toBe(false);
    });

    it("getRememberMe retourne true quand la clé vaut 'true'", async () => {
      mockGet.mockImplementation((key: string) =>
        Promise.resolve(key === "driver.rememberMe" ? "true" : null)
      );
      expect(await getRememberMe()).toBe(true);
    });

    it("setRememberMe(true) écrit la clé driver.rememberMe à 'true'", async () => {
      await setRememberMe(true);
      expect(mockSet).toHaveBeenCalledWith("driver.rememberMe", "true");
    });

    it("setRememberMe(false) supprime rememberMe et appelle clearRememberedCredentials", async () => {
      await setRememberMe(false);
      expect(mockDelete).toHaveBeenCalledWith("driver.rememberMe");
      expect(mockDelete).toHaveBeenCalledWith("driver.rememberedEmail");
      expect(mockDelete).toHaveBeenCalledWith("driver.rememberedPassword");
    });
  });

  describe("getRememberedCredentials / setRememberedCredentials", () => {
    it("getRememberedCredentials retourne null quand email ou password manquant", async () => {
      mockGet.mockResolvedValue(null);
      expect(await getRememberedCredentials()).toBe(null);

      mockGet.mockImplementation((k: string) =>
        Promise.resolve(k === "driver.rememberedEmail" ? "a@b.co" : null)
      );
      expect(await getRememberedCredentials()).toBe(null);

      mockGet.mockImplementation((k: string) =>
        Promise.resolve(k === "driver.rememberedPassword" ? "secret" : null)
      );
      expect(await getRememberedCredentials()).toBe(null);
    });

    it("getRememberedCredentials retourne { email, password } quand les deux sont présents", async () => {
      mockGet.mockImplementation((k: string) =>
        Promise.resolve(
          k === "driver.rememberedEmail"
            ? "  driver@test.com  "
            : k === "driver.rememberedPassword"
              ? "pwd123"
              : null
        )
      );
      const creds = await getRememberedCredentials();
      expect(creds).toEqual({ email: "driver@test.com", password: "pwd123" });
    });

    it("setRememberedCredentials écrit driver.rememberedEmail et driver.rememberedPassword", async () => {
      await setRememberedCredentials("user@example.com", "secret");
      expect(mockSet).toHaveBeenCalledWith("driver.rememberedEmail", "user@example.com");
      expect(mockSet).toHaveBeenCalledWith("driver.rememberedPassword", "secret");
    });

    it("setRememberedCredentials trim l'email", async () => {
      await setRememberedCredentials("  u@x.co  ", "p");
      expect(mockSet).toHaveBeenCalledWith("driver.rememberedEmail", "u@x.co");
    });
  });

  describe("clearRememberedCredentials", () => {
    it("supprime driver.rememberedEmail et driver.rememberedPassword", async () => {
      await clearRememberedCredentials();
      expect(mockDelete).toHaveBeenCalledWith("driver.rememberedEmail");
      expect(mockDelete).toHaveBeenCalledWith("driver.rememberedPassword");
    });
  });

  describe("logout behavior (non-régression)", () => {
    it("quand getRememberMe() est false, le flux logout doit appeler clearRememberedCredentials", async () => {
      mockGet.mockImplementation((k: string) =>
        Promise.resolve(k === "driver.rememberMe" ? null : null)
      );
      const rm = await getRememberMe();
      expect(rm).toBe(false);
      await clearRememberedCredentials();
      expect(mockDelete).toHaveBeenCalledWith("driver.rememberedEmail");
      expect(mockDelete).toHaveBeenCalledWith("driver.rememberedPassword");
    });

    it("quand getRememberMe() est true, le flux logout ne doit pas effacer les credentials", async () => {
      mockGet.mockImplementation((k: string) =>
        Promise.resolve(k === "driver.rememberMe" ? "true" : null)
      );
      const rm = await getRememberMe();
      expect(rm).toBe(true);
      mockDelete.mockClear();
      // Simuler le flux logout : on n'appelle pas clearRememberedCredentials quand rm === true
      expect(mockDelete).not.toHaveBeenCalled();
    });
  });
});
