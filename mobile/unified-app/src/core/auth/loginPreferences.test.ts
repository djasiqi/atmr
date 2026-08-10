import { beforeEach, describe, expect, it, jest } from "@jest/globals";

import {
  persistLoginRememberMe,
  readLoginPreferences,
  writeLoginPreferences,
} from "./loginPreferences";
import { STORAGE_KEYS } from "../storage/storageKeys";

jest.mock("../storage/typedStorage", () => ({
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn(),
}));

jest.mock("../storage/secureStoreCompat", () => ({
  getItemAsync: jest.fn(),
  setItemAsync: jest.fn(),
  deleteItemAsync: jest.fn(),
}));

const { getItem, setItem, removeItem } = jest.requireMock<{
  getItem: jest.Mock;
  setItem: jest.Mock;
  removeItem: jest.Mock;
}>("../storage/typedStorage");

const SecureStore = jest.requireMock<{
  getItemAsync: jest.Mock;
  setItemAsync: jest.Mock;
  deleteItemAsync: jest.Mock;
}>("../storage/secureStoreCompat");

describe("loginPreferences", () => {
  beforeEach(() => {
    getItem.mockReset();
    setItem.mockReset();
    removeItem.mockReset();
    SecureStore.getItemAsync.mockReset();
    SecureStore.setItemAsync.mockReset();
    SecureStore.deleteItemAsync.mockReset();
  });

  it("readLoginPreferences renvoie les valeurs par défaut si absent", async () => {
    getItem.mockResolvedValue(null);
    await expect(readLoginPreferences()).resolves.toEqual({
      rememberMe: true,
      email: null,
      password: null,
    });
    expect(SecureStore.deleteItemAsync).toHaveBeenCalledWith(
      STORAGE_KEYS.LOGIN_REMEMBERED_PASSWORD
    );
  });

  it("readLoginPreferences charge l'email sans jamais renvoyer de mot de passe", async () => {
    getItem.mockResolvedValue({
      rememberMe: true,
      email: "user@example.com",
    });
    await expect(readLoginPreferences()).resolves.toEqual({
      rememberMe: true,
      email: "user@example.com",
      password: null,
    });
    expect(SecureStore.deleteItemAsync).toHaveBeenCalledWith(
      STORAGE_KEYS.LOGIN_REMEMBERED_PASSWORD
    );
  });

  it("persistLoginRememberMe enregistre l'email sans mot de passe", async () => {
    await persistLoginRememberMe("user@example.com", "secret123", true);
    expect(setItem).toHaveBeenCalledWith(STORAGE_KEYS.LOGIN_PREFERENCES, {
      rememberMe: true,
      email: "user@example.com",
    });
    expect(SecureStore.setItemAsync).not.toHaveBeenCalled();
    expect(SecureStore.deleteItemAsync).toHaveBeenCalledWith(
      STORAGE_KEYS.LOGIN_REMEMBERED_PASSWORD
    );
  });

  it("writeLoginPreferences supprime email et mot de passe si rememberMe est false", async () => {
    await writeLoginPreferences({ rememberMe: false, email: null, password: null });
    expect(removeItem).toHaveBeenCalledWith(STORAGE_KEYS.LOGIN_PREFERENCES);
    expect(SecureStore.deleteItemAsync).toHaveBeenCalledWith(
      STORAGE_KEYS.LOGIN_REMEMBERED_PASSWORD
    );
  });
});
