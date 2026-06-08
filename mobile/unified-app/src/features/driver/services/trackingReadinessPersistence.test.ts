import {
  markTrackingOnboarded,
  readTrackingNeedsAttention,
  readTrackingOnboarded,
  resetTrackingOnboarded,
  setTrackingNeedsAttention,
} from "./trackingReadinessPersistence";

jest.mock("../../../core/storage/typedStorage", () => ({
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn(),
}));

const storage = jest.requireMock("../../../core/storage/typedStorage") as {
  getItem: jest.Mock;
  setItem: jest.Mock;
  removeItem: jest.Mock;
};

describe("trackingReadinessPersistence", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    storage.getItem.mockResolvedValue(null);
  });

  it("trackingNeedsAttention set/clear sans reset onboarded", async () => {
    storage.getItem.mockImplementation(async (key: string) => {
      if (key === "driver.tracking_onboarded") {
        return { onboarded: true, at: "2026-01-01T00:00:00.000Z" };
      }
      if (key === "driver.tracking_needs_attention") {
        return { needsAttention: true, at: "2026-01-02T00:00:00.000Z" };
      }
      return null;
    });

    expect(await readTrackingOnboarded()).toBe(true);
    expect(await readTrackingNeedsAttention()).toBe(true);

    await setTrackingNeedsAttention(false);
    expect(storage.removeItem).toHaveBeenCalledWith("driver.tracking_needs_attention");
    expect(storage.removeItem).not.toHaveBeenCalledWith("driver.tracking_onboarded");

    await setTrackingNeedsAttention(true);
    expect(storage.setItem).toHaveBeenCalledWith(
      "driver.tracking_needs_attention",
      expect.objectContaining({ needsAttention: true })
    );

    await resetTrackingOnboarded();
    expect(storage.removeItem).toHaveBeenCalledWith("driver.tracking_onboarded");
  });

  it("markTrackingOnboarded persiste onboarded", async () => {
    await markTrackingOnboarded();
    expect(storage.setItem).toHaveBeenCalledWith(
      "driver.tracking_onboarded",
      expect.objectContaining({ onboarded: true })
    );
  });
});
