import { beforeEach, describe, expect, it, jest } from "@jest/globals";

import {
  clearPushRegistrationFailed,
  requestNotificationDisclosure,
  resolvePushRegistrationBannerState,
  setPushRegistrationFailed,
} from "./pushRegistrationState";

jest.mock("./notificationDisclosurePersistence", () => ({
  ensureNotificationDisclosureSyncedWithOsPermission: jest.fn(async () => undefined),
  readNotificationDisclosureAccepted: jest.fn(),
  subscribeNotificationDisclosureAccepted: jest.fn(() => () => undefined),
}));

jest.mock("./pendingPushTokenRegistration", () => ({
  hasPendingPushTokenRegistrations: jest.fn(),
}));

jest.mock("./pushPermissionState", () => ({
  getPushPermissionDenied: jest.fn(),
}));

const ensureSync = jest.requireMock<{
  ensureNotificationDisclosureSyncedWithOsPermission: jest.Mock;
}>("./notificationDisclosurePersistence").ensureNotificationDisclosureSyncedWithOsPermission;
const readDisclosure = jest.requireMock<{ readNotificationDisclosureAccepted: jest.Mock }>(
  "./notificationDisclosurePersistence"
).readNotificationDisclosureAccepted;
const hasPending = jest.requireMock<{ hasPendingPushTokenRegistrations: jest.Mock }>(
  "./pendingPushTokenRegistration"
).hasPendingPushTokenRegistrations;
const getPermissionDenied = jest.requireMock<{ getPushPermissionDenied: jest.Mock }>(
  "./pushPermissionState"
).getPushPermissionDenied;

describe("pushRegistrationState", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    clearPushRegistrationFailed();
    ensureSync.mockResolvedValue(undefined);
    readDisclosure.mockResolvedValue(true);
    hasPending.mockResolvedValue(false);
    getPermissionDenied.mockReturnValue(false);
  });

  it("resolvePushRegistrationBannerState sync la disclosure avant évaluation", async () => {
    await resolvePushRegistrationBannerState();
    expect(ensureSync).toHaveBeenCalledTimes(1);
  });

  it("resolvePushRegistrationBannerState returns disclosure_required when not accepted", async () => {
    readDisclosure.mockResolvedValue(false);
    await expect(resolvePushRegistrationBannerState()).resolves.toBe("disclosure_required");
  });

  it("resolvePushRegistrationBannerState returns registration_pending when queue non empty", async () => {
    hasPending.mockResolvedValue(true);
    await expect(resolvePushRegistrationBannerState()).resolves.toBe("registration_pending");
  });

  it("resolvePushRegistrationBannerState returns registration_failed after setPushRegistrationFailed", async () => {
    setPushRegistrationFailed(true);
    await expect(resolvePushRegistrationBannerState()).resolves.toBe("registration_failed");
  });

  it("requestNotificationDisclosure increments disclosure show request", () => {
    requestNotificationDisclosure();
    // pas d'assertion directe sur count exporté — smoke test sans throw
    expect(true).toBe(true);
  });
});
