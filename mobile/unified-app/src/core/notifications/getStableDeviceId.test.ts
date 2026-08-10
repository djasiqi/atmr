import {
  readInstallationId,
  createAndPersistInstallationId,
} from "../auth/authCredentialStore";
import { getStableDeviceId, resetStableDeviceIdCacheForTests } from "./getStableDeviceId";

jest.mock("expo-secure-store", () => ({
  getItemAsync: jest.fn(),
  setItemAsync: jest.fn(),
}));

jest.mock("../auth/authCredentialStore", () => ({
  readInstallationId: jest.fn(),
  createAndPersistInstallationId: jest.fn(),
}));

const mockReadInstallationId = readInstallationId as jest.MockedFunction<
  typeof readInstallationId
>;
const mockCreateAndPersistInstallationId =
  createAndPersistInstallationId as jest.MockedFunction<
    typeof createAndPersistInstallationId
  >;

describe("getStableDeviceId", () => {
  beforeEach(() => {
    resetStableDeviceIdCacheForTests();
    jest.clearAllMocks();
  });

  it("retourne l'ID SecureStore existant et le met en cache", async () => {
    mockReadInstallationId.mockResolvedValue({ status: "found", value: "stored-device-id" });
    const id = await getStableDeviceId();
    expect(id).toBe("stored-device-id");
    const again = await getStableDeviceId();
    expect(again).toBe("stored-device-id");
    expect(mockReadInstallationId).toHaveBeenCalledTimes(1);
  });

  it("crée un ID si aucun n'existe", async () => {
    mockReadInstallationId.mockResolvedValue({ status: "missing" });
    mockCreateAndPersistInstallationId.mockResolvedValue({
      status: "found",
      value: "atmr-new-id",
    });
    const id = await getStableDeviceId();
    expect(id).toBe("atmr-new-id");
    expect(mockCreateAndPersistInstallationId).toHaveBeenCalled();
  });

  it("échoue si SecureStore est temporairement indisponible", async () => {
    mockReadInstallationId.mockResolvedValue({ status: "temporarily_unavailable" });
    await expect(getStableDeviceId()).rejects.toThrow(
      "device_identity_storage_unavailable"
    );
    expect(mockCreateAndPersistInstallationId).not.toHaveBeenCalled();
  });
});
