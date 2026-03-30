import AsyncStorage from "@react-native-async-storage/async-storage";
import {
  clearLocationQueue,
  enqueueLocation,
  flushLatestPositionViaHttp,
  recordDriverLocationBatchAckSuccess,
} from "../locationQueue";
import { getSocket, getSocketRole } from "../socket";
import { getNetworkStateSnapshot } from "../networkState";

jest.mock("@react-native-async-storage/async-storage", () => ({
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn(),
}));

jest.mock("../socket", () => ({
  getSocket: jest.fn(),
  getSocketRole: jest.fn(),
}));

jest.mock("../networkState", () => ({
  getNetworkStateSnapshot: jest.fn(),
}));

jest.mock("../api", () => ({
  updateDriverLocation: jest.fn().mockResolvedValue({ ok: true }),
}));

const store = new Map<string, string>();

beforeEach(() => {
  jest.clearAllMocks();
  store.clear();
  (AsyncStorage.getItem as jest.Mock).mockImplementation(async (key: string) => {
    return store.get(key) ?? null;
  });
  (AsyncStorage.setItem as jest.Mock).mockImplementation(async (key: string, value: string) => {
    store.set(key, value);
  });
  (getSocketRole as jest.Mock).mockReturnValue("driver");
  (getNetworkStateSnapshot as jest.Mock).mockReturnValue({
    isConnected: true,
    isInternetReachable: true,
  });
  (getSocket as jest.Mock).mockReturnValue({
    connected: true,
    auth: { token: "tok" },
  });
});

describe("flushLatestPositionViaHttp", () => {
  it("batch pipeline sain + point mission => 0 PUT HTTP", async () => {
    const { updateDriverLocation } = require("../api");
    (updateDriverLocation as jest.Mock).mockClear();

    await clearLocationQueue();
    recordDriverLocationBatchAckSuccess();
    await enqueueLocation({
      driver_id: 1,
      latitude: 46.2,
      longitude: 6.1,
      speed: 0,
      heading: 0,
      accuracy: 10,
      timestamp: Date.now(),
      location_mode: "mission_live",
    });

    await flushLatestPositionViaHttp();

    expect(updateDriverLocation).not.toHaveBeenCalled();
  });

  it("availability_presence => PUT HTTP conservé (batch socket exclut la présence)", async () => {
    const { updateDriverLocation } = require("../api");
    (updateDriverLocation as jest.Mock).mockClear();

    await clearLocationQueue();
    recordDriverLocationBatchAckSuccess();
    await enqueueLocation({
      driver_id: 1,
      latitude: 46.2,
      longitude: 6.1,
      speed: 0,
      heading: 0,
      accuracy: 10,
      timestamp: Date.now(),
      location_mode: "availability_presence",
    });

    await flushLatestPositionViaHttp();

    expect(updateDriverLocation).toHaveBeenCalledTimes(1);
  });

  it("socket non opérationnel => PUT HTTP maintenu", async () => {
    const { updateDriverLocation } = require("../api");
    (updateDriverLocation as jest.Mock).mockClear();
    (getSocket as jest.Mock).mockReturnValue(null);

    await clearLocationQueue();
    await enqueueLocation({
      driver_id: 1,
      latitude: 46.2,
      longitude: 6.1,
      speed: 0,
      heading: 0,
      accuracy: 10,
      timestamp: Date.now(),
      location_mode: "mission_live",
    });

    await flushLatestPositionViaHttp();

    expect(updateDriverLocation).toHaveBeenCalledTimes(1);
  });
});
