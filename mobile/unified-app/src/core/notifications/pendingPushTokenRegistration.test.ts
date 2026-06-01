import { beforeEach, describe, expect, it, jest } from "@jest/globals";

import { STORAGE_KEYS } from "../storage/storageKeys";
import {
  clearPendingPushTokenRegistration,
  flushPendingPushTokenRegistrations,
  persistPendingPushTokenRegistration,
  registerWithRetry,
} from "./pendingPushTokenRegistration";

const mockGetItem = jest.fn() as jest.Mock<any>;
const mockSetItem = jest.fn() as jest.Mock<any>;
const mockRemoveItem = jest.fn() as jest.Mock<any>;

jest.mock("../storage/typedStorage", () => ({
  getItem: (...args: unknown[]) => mockGetItem(...args),
  setItem: (...args: unknown[]) => mockSetItem(...args),
  removeItem: (...args: unknown[]) => mockRemoveItem(...args),
}));

describe("pendingPushTokenRegistration", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockGetItem.mockResolvedValue(null);
    mockSetItem.mockResolvedValue(undefined);
    mockRemoveItem.mockResolvedValue(undefined);
  });

  it("registerWithRetry succeeds on second attempt", async () => {
    const fn = jest
      .fn<() => Promise<void>>()
      .mockRejectedValueOnce(new Error("network"))
      .mockResolvedValue(undefined);

    await registerWithRetry(fn, 3);

    expect(fn).toHaveBeenCalledTimes(2);
  });

  it("registerWithRetry throws after max attempts", async () => {
    const fn = jest.fn<() => Promise<void>>().mockRejectedValue(new Error("network"));

    await expect(registerWithRetry(fn, 2)).rejects.toThrow("network");
    expect(fn).toHaveBeenCalledTimes(2);
  });

  it("persistPendingPushTokenRegistration stores by provider", async () => {
    await persistPendingPushTokenRegistration({
      provider: "fcm",
      token: "tok",
      deviceId: "dev",
      platform: "android",
    });

    expect(mockSetItem).toHaveBeenCalledWith(
      STORAGE_KEYS.PENDING_PUSH_TOKEN_REGISTRATION,
      expect.objectContaining({
        items: [
          expect.objectContaining({
            provider: "fcm",
            token: "tok",
          }),
        ],
      })
    );
  });

  it("flushPendingPushTokenRegistrations clears item on success", async () => {
    mockGetItem.mockResolvedValue({
      items: [
        {
          provider: "expo",
          token: "expo-tok",
          deviceId: "dev",
          platform: "ios",
          savedAt: Date.now(),
        },
      ],
    });
    const registerExpo = jest.fn<() => Promise<void>>().mockResolvedValue(undefined);
    const registerFcm = jest.fn<() => Promise<void>>().mockResolvedValue(undefined);

    await flushPendingPushTokenRegistrations({ registerExpo, registerFcm });

    expect(registerExpo).toHaveBeenCalledTimes(1);
    expect(mockRemoveItem).toHaveBeenCalledWith(STORAGE_KEYS.PENDING_PUSH_TOKEN_REGISTRATION);
  });

  it("clearPendingPushTokenRegistration removes only matching provider", async () => {
    mockGetItem.mockResolvedValue({
      items: [
        {
          provider: "expo",
          token: "a",
          deviceId: "d",
          platform: "android",
          savedAt: 1,
        },
        {
          provider: "fcm",
          token: "b",
          deviceId: "d",
          platform: "android",
          savedAt: 2,
        },
      ],
    });

    await clearPendingPushTokenRegistration("expo");

    expect(mockSetItem).toHaveBeenCalledWith(
      STORAGE_KEYS.PENDING_PUSH_TOKEN_REGISTRATION,
      expect.objectContaining({
        items: [expect.objectContaining({ provider: "fcm" })],
      })
    );
  });
});
