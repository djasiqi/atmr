import { getInstallationIdAsync } from "expo-application";
import * as SecureStore from "expo-secure-store";

import { getStableDeviceId, resetStableDeviceIdCacheForTests } from "./getStableDeviceId";

jest.mock("expo-secure-store", () => ({
  getItemAsync: jest.fn(),
  setItemAsync: jest.fn(),
}));

jest.mock("expo-application", () => ({
  getInstallationIdAsync: jest.fn().mockResolvedValue("install-abc-123"),
}));

describe("getStableDeviceId", () => {
  beforeEach(() => {
    resetStableDeviceIdCacheForTests();
    jest.clearAllMocks();
  });

  it("returns installation id when available", async () => {
    const id = await getStableDeviceId();
    expect(id).toBe("install-abc-123");
    const again = await getStableDeviceId();
    expect(again).toBe("install-abc-123");
  });

  it("falls back to SecureStore when installation id unavailable", async () => {
    jest.mocked(getInstallationIdAsync).mockResolvedValueOnce(null);
    (SecureStore.getItemAsync as jest.Mock).mockResolvedValueOnce("stored-device-id");

    resetStableDeviceIdCacheForTests();
    const id = await getStableDeviceId();
    expect(id).toBe("stored-device-id");
  });
});
