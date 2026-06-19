import { beforeEach, describe, expect, it, jest } from "@jest/globals";

import {
  ensureNotificationDisclosureSyncedWithOsPermission,
  markNotificationDisclosureAccepted,
  readNotificationDisclosureAccepted,
  resetNotificationDisclosureAccepted,
} from "./notificationDisclosurePersistence";

jest.mock("./expoNotificationsCompat", () => ({
  getExpoNotificationsModule: jest.fn(),
}));

jest.mock("../storage/typedStorage", () => ({
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn(),
}));

const getExpoNotificationsModule = jest.requireMock<{ getExpoNotificationsModule: jest.Mock }>(
  "./expoNotificationsCompat"
).getExpoNotificationsModule;
const { getItem, setItem } = jest.requireMock<{
  getItem: jest.Mock;
  setItem: jest.Mock;
}>("../storage/typedStorage");

describe("ensureNotificationDisclosureSyncedWithOsPermission", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    getItem.mockResolvedValue(null);
    setItem.mockResolvedValue(undefined);
    getExpoNotificationsModule.mockReturnValue({
      getPermissionsAsync: jest.fn().mockResolvedValue({ granted: false, status: "denied" }),
    });
  });

  it("ne fait rien si la disclosure est déjà acceptée", async () => {
    getItem.mockResolvedValue({ accepted: true, at: "2026-01-01T00:00:00.000Z" });

    await expect(ensureNotificationDisclosureSyncedWithOsPermission()).resolves.toBe(true);
    expect(setItem).not.toHaveBeenCalled();
  });

  it("marque la disclosure acceptée si la permission OS est déjà accordée", async () => {
    getExpoNotificationsModule.mockReturnValue({
      getPermissionsAsync: jest.fn().mockResolvedValue({ granted: true, status: "granted" }),
    });

    await expect(ensureNotificationDisclosureSyncedWithOsPermission()).resolves.toBe(true);
    expect(setItem).toHaveBeenCalledTimes(1);
  });

  it("laisse la disclosure non acceptée si la permission OS est refusée", async () => {
    await expect(ensureNotificationDisclosureSyncedWithOsPermission()).resolves.toBe(false);
    expect(setItem).not.toHaveBeenCalled();
  });

  it("markNotificationDisclosureAccepted persiste l'acceptation", async () => {
    await resetNotificationDisclosureAccepted();
    await markNotificationDisclosureAccepted();
    expect(setItem).toHaveBeenCalled();
  });
});
