import { describe, expect, it, jest } from "@jest/globals";
import {
  DRIVER_NOTIFICATION_ACTION_IDS,
  DRIVER_NOTIFICATION_CATEGORY_ID,
  ensureDriverNotificationActions,
} from "./notificationActions";

const mockSetNotificationCategoryAsync = jest.fn() as jest.Mock<any>;

jest.mock("expo-notifications", () => ({
  setNotificationCategoryAsync: (...args: unknown[]) =>
    mockSetNotificationCategoryAsync(...args),
}));

describe("ensureDriverNotificationActions", () => {
  it("registers mission category with action identifiers", async () => {
    mockSetNotificationCategoryAsync.mockResolvedValue(undefined);
    await ensureDriverNotificationActions();

    expect(mockSetNotificationCategoryAsync).toHaveBeenCalledWith(
      DRIVER_NOTIFICATION_CATEGORY_ID,
      expect.arrayContaining([
        expect.objectContaining({
          identifier: DRIVER_NOTIFICATION_ACTION_IDS.ACCEPT,
          options: expect.objectContaining({ opensAppToForeground: false }),
        }),
        expect.objectContaining({
          identifier: DRIVER_NOTIFICATION_ACTION_IDS.DECLINE,
          options: expect.objectContaining({
            isDestructive: true,
            opensAppToForeground: false,
          }),
        }),
      ])
    );
  });
});
