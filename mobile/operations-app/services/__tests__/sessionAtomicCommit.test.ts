import * as SecureStore from "expo-secure-store";
import AsyncStorage from "@react-native-async-storage/async-storage";

import {
  commitSessionTokensAtomically,
  secureStorage,
  setActiveAuthNamespace,
} from "../storage";

jest.mock("expo-secure-store", () => ({
  setItemAsync: jest.fn(),
  getItemAsync: jest.fn(),
  deleteItemAsync: jest.fn(),
  AFTER_FIRST_UNLOCK: "AFTER_FIRST_UNLOCK",
}));

jest.mock("@react-native-async-storage/async-storage", () => ({
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn(),
  multiRemove: jest.fn(),
}));

describe("commitSessionTokensAtomically", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    (SecureStore.setItemAsync as jest.Mock).mockResolvedValue(undefined);
    (AsyncStorage.setItem as jest.Mock).mockResolvedValue(undefined);
  });

  it("commit driver access+refresh", async () => {
    await commitSessionTokensAtomically({
      scope: "driver",
      accessToken: "access-1",
      refreshToken: "refresh-1",
      trigger_source: "api_interceptor",
    });

    expect(SecureStore.setItemAsync).toHaveBeenCalled();
    expect(await secureStorage.getAccessToken()).toBe("access-1");
  });

  it("throw en cas d'échec partiel de persistance", async () => {
    (SecureStore.setItemAsync as jest.Mock).mockRejectedValueOnce(
      new Error("securestore_down")
    );

    await expect(
      commitSessionTokensAtomically({
        scope: "enterprise",
        accessToken: "ent-access",
        refreshToken: "ent-refresh",
        sessionStorageKey: "enterprise.session",
        sessionMeta: { companyId: 1 },
        trigger_source: "api_interceptor",
      })
    ).rejects.toThrow("securestore_down");
  });

  it("migre une clé legacy vers namespace actif", async () => {
    await secureStorage.removeAccessToken();
    (AsyncStorage.setItem as jest.Mock).mockResolvedValue(undefined);
    (AsyncStorage.getItem as jest.Mock).mockImplementation(async (key: string) => {
      if (key === "auth.namespace.driver") {
        return "driver:user-123:none:none";
      }
      return null;
    });
    (SecureStore.getItemAsync as jest.Mock).mockImplementation(async (key: string) => {
      if (key === "driver_access_token:driver:user-123:none:none") {
        return null;
      }
      if (key === "driver_access_token") {
        return "legacy-access-token";
      }
      return null;
    });

    await setActiveAuthNamespace({
      role: "driver",
      userId: "user-123",
      tenantId: null,
      sessionId: null,
    });
    const token = await secureStorage.getAccessToken();

    expect(token).toBe("legacy-access-token");
    expect(SecureStore.setItemAsync).toHaveBeenCalledWith(
      "driver_access_token.ns.driver.user-123.none.none",
      "legacy-access-token",
      expect.anything()
    );
  });
});
