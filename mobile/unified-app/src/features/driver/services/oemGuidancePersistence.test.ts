import {
  clearOemGuidanceAcknowledgement,
  isOemGuidanceAcknowledgedFor,
  markOemGuidanceAcknowledged,
  readOemGuidanceAcknowledgement,
} from "./oemGuidancePersistence";

jest.mock("../../../core/storage/typedStorage", () => ({
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn(),
}));

jest.mock("expo-constants", () => ({
  __esModule: true,
  default: { expoConfig: { version: "1.0.10" }, nativeAppVersion: "1.0.10" },
}));

const storage = jest.requireMock("../../../core/storage/typedStorage") as {
  getItem: jest.Mock;
  setItem: jest.Mock;
  removeItem: jest.Mock;
};

describe("oemGuidancePersistence", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    storage.getItem.mockResolvedValue(null);
  });

  it("markOemGuidanceAcknowledged persiste oem + appVersion", async () => {
    await markOemGuidanceAcknowledged("samsung");
    expect(storage.setItem).toHaveBeenCalledWith(
      "driver.oem_guidance_acknowledged",
      expect.objectContaining({
        oem: "samsung",
        appVersion: "1.0.10",
        acknowledgedAt: expect.any(String),
      })
    );
  });

  it("acquittement valide seulement pour le même OEM", async () => {
    storage.getItem.mockResolvedValue({
      acknowledgedAt: "2026-01-01T00:00:00.000Z",
      oem: "samsung",
    });
    expect(await isOemGuidanceAcknowledgedFor("samsung")).toBe(true);
    expect(await isOemGuidanceAcknowledgedFor("xiaomi")).toBe(false);
    expect(await isOemGuidanceAcknowledgedFor(null)).toBe(false);
  });

  it("read / clear", async () => {
    storage.getItem.mockResolvedValue({
      acknowledgedAt: "2026-01-01T00:00:00.000Z",
      oem: "samsung",
    });
    expect(await readOemGuidanceAcknowledgement()).toEqual(
      expect.objectContaining({ oem: "samsung" })
    );
    await clearOemGuidanceAcknowledgement();
    expect(storage.removeItem).toHaveBeenCalledWith("driver.oem_guidance_acknowledged");
  });
});
