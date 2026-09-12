import { beforeEach, describe, expect, it, jest } from "@jest/globals";
import { Platform } from "react-native";
import * as NativeSecureStore from "expo-secure-store";

import * as SecureStoreCompat from "./secureStoreCompat";

describe("secureStoreCompat", () => {
  beforeEach(async () => {
    await SecureStoreCompat.deleteItemAsync("revocation_secret");
    await SecureStoreCompat.deleteItemAsync("atmr.auth.refresh_token");
    if (typeof localStorage !== "undefined") {
      localStorage.clear();
      localStorage.setItem("unified_secure_store:revocation_secret", "legacy-secret");
    }
  });

  it("web : mémoire seule, jamais localStorage", async () => {
    const originalOs = Platform.OS;
    Object.defineProperty(Platform, "OS", { configurable: true, value: "web" });

    await SecureStoreCompat.setItemAsync("revocation_secret", "runtime-secret");
    await expect(SecureStoreCompat.getItemAsync("revocation_secret")).resolves.toBe(
      "runtime-secret"
    );

    if (typeof localStorage !== "undefined") {
      expect(localStorage.getItem("unified_secure_store:revocation_secret")).toBeNull();
      for (let i = 0; i < localStorage.length; i += 1) {
        const key = localStorage.key(i);
        expect(key).not.toMatch(/^unified_secure_store:/);
      }
    }

    Object.defineProperty(Platform, "OS", { configurable: true, value: originalOs });
  });

  it("web : un secret legacy en localStorage n'est pas réinjecté", async () => {
    const originalOs = Platform.OS;
    Object.defineProperty(Platform, "OS", { configurable: true, value: "web" });

    await expect(SecureStoreCompat.getItemAsync("revocation_secret")).resolves.toBeNull();
    if (typeof localStorage !== "undefined") {
      expect(localStorage.getItem("unified_secure_store:revocation_secret")).toBeNull();
    }

    Object.defineProperty(Platform, "OS", { configurable: true, value: originalOs });
  });

  it("native : délègue à expo-secure-store sans localStorage", async () => {
    const originalOs = Platform.OS;
    Object.defineProperty(Platform, "OS", { configurable: true, value: "ios" });

    const setSpy = NativeSecureStore.setItemAsync as jest.Mock;
    setSpy.mockClear();
    setSpy.mockResolvedValue(undefined);

    await SecureStoreCompat.setItemAsync("atmr.auth.refresh_token", "native-refresh");
    expect(setSpy).toHaveBeenCalledWith("atmr.auth.refresh_token", "native-refresh");

    if (typeof localStorage !== "undefined") {
      expect(localStorage.getItem("unified_secure_store:atmr.auth.refresh_token")).toBeNull();
    }

    Object.defineProperty(Platform, "OS", { configurable: true, value: originalOs });
  });
});
