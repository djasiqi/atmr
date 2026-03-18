import AsyncStorage from "@react-native-async-storage/async-storage";
import {
  clearLocationQueue,
  enqueueLocation,
  syncLocationQueue,
} from "../locationQueue";
import { getSocket, getSocketRole } from "../socket";

jest.mock("@react-native-async-storage/async-storage", () => ({
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn(),
}));

jest.mock("../socket", () => ({
  getSocket: jest.fn(),
  getSocketRole: jest.fn(),
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
  (AsyncStorage.removeItem as jest.Mock).mockImplementation(async (key: string) => {
    store.delete(key);
  });
  (getSocketRole as jest.Mock).mockReturnValue("driver");
});

describe("locationQueue socket presence guards", () => {
  it("fallback individuel envoie un payload complet (mode + timestamps)", async () => {
    const individualPayloads: any[] = [];
    const socket = {
      connected: true,
      emit: jest.fn((event: string, payload: any, ack?: (data: any) => void) => {
        if (event === "driver_location_batch" && typeof ack === "function") {
          ack({ success: false, error: "forced ack failure" });
          return;
        }
        if (event === "driver_location") {
          individualPayloads.push(payload);
        }
      }),
    };
    (getSocket as jest.Mock).mockReturnValue(socket);

    await clearLocationQueue();
    await enqueueLocation({
      driver_id: 7,
      latitude: 46.2,
      longitude: 6.1,
      speed: 9,
      heading: 120,
      accuracy: 6,
      timestamp: Date.now(),
      location_mode: "mission_live",
    });

    // 3 échecs batch pour déclencher le fallback individuel au 4e appel.
    await expect(syncLocationQueue(socket)).rejects.toBeTruthy();
    await expect(syncLocationQueue(socket)).rejects.toBeTruthy();
    await expect(syncLocationQueue(socket)).rejects.toBeTruthy();
    await syncLocationQueue(socket);

    expect(individualPayloads.length).toBeGreaterThan(0);
    const payload = individualPayloads[0];
    expect(payload.location_mode).toBe("mission_live");
    expect(typeof payload.recorded_at).toBe("string");
    expect(typeof payload.sent_at).toBe("string");
  });

});
