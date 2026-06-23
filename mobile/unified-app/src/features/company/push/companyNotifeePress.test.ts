import { describe, expect, it, jest } from "@jest/globals";

import {
  consumePendingCompanyPushPress,
  isCompanyInstitutionPushPayload,
  persistPendingCompanyPushPress,
} from "./companyNotifeePress";

jest.mock("@react-native-async-storage/async-storage", () => ({
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn(),
}));

const storage = jest.requireMock<{
  getItem: jest.Mock;
  setItem: jest.Mock;
  removeItem: jest.Mock;
}>("@react-native-async-storage/async-storage");

describe("isCompanyInstitutionPushPayload", () => {
  it("reconnaît new_request", () => {
    expect(isCompanyInstitutionPushPayload({ type: "new_request", offer_id: 1 })).toBe(true);
  });

  it("ignore les payloads chauffeur", () => {
    expect(isCompanyInstitutionPushPayload({ type: "booking_assigned", booking_id: 1 })).toBe(
      false
    );
  });
});

describe("pending company push press", () => {
  beforeEach(() => {
    storage.getItem.mockReset();
    storage.setItem.mockReset();
    storage.removeItem.mockReset();
  });

  it("persiste puis consomme le tap", async () => {
    const data = { type: "new_request", offer_id: "42" };
    await persistPendingCompanyPushPress(data);
    expect(storage.setItem).toHaveBeenCalled();

    storage.getItem.mockResolvedValueOnce(
      JSON.stringify({ data, savedAt: Date.now() })
    );
    const consumed = await consumePendingCompanyPushPress();
    expect(consumed).toEqual(data);
    expect(storage.removeItem).toHaveBeenCalled();
  });
});
