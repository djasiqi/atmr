import AsyncStorage from "@react-native-async-storage/async-storage";
import { apiClient } from "../../../core/api/client";
import { geocodeMissionAddress } from "./geocodeMissionAddress";

jest.mock("@react-native-async-storage/async-storage", () => ({
  getItem: jest.fn().mockResolvedValue(null),
  setItem: jest.fn().mockResolvedValue(undefined),
  removeItem: jest.fn().mockResolvedValue(undefined),
}));

jest.mock("../../../core/api/client", () => ({
  apiClient: { get: jest.fn() },
}));

describe("geocodeMissionAddress", () => {
  const fetchMock = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    (globalThis as { fetch?: typeof fetch }).fetch = fetchMock as unknown as typeof fetch;
  });

  it("demande les coordonnées au backend et n'appelle pas Google", async () => {
    (apiClient.get as jest.Mock).mockResolvedValue({
      data: { lat: 46.2, lon: 6.14 },
    });

    const coord = await geocodeMissionAddress("Rue de la Navigation 1, Genève");

    expect(coord).toEqual({ lat: 46.2, lng: 6.14 });
    expect(apiClient.get).toHaveBeenCalledWith("/geocode/geocode", {
      params: { address: "Rue de la Navigation 1, Genève", country: "CH" },
    });
    expect(fetchMock).not.toHaveBeenCalled();
    expect(AsyncStorage.setItem).toHaveBeenCalled();
  });

  it("retourne null si le backend ne répond pas", async () => {
    (apiClient.get as jest.Mock).mockRejectedValue(new Error("réseau"));

    await expect(geocodeMissionAddress("Adresse inconnue")).resolves.toBeNull();
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
