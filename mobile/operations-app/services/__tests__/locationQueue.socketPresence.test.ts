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
    // 4e appel : fallback déclenché → throw "Fallback sent without ACK" (comportement attendu)
    await expect(syncLocationQueue(socket)).rejects.toThrow("Fallback sent without ACK");

    expect(individualPayloads.length).toBeGreaterThan(0);
    const payload = individualPayloads[0];
    expect(payload.location_mode).toBe("mission_live");
    expect(typeof payload.recorded_at).toBe("string");
    expect(typeof payload.sent_at).toBe("string");
  });

  it("emitBatchWithAck normalise les positions legacy sans location_mode/recorded_at", async () => {
    const emittedBatches: any[] = [];
    const socket = {
      connected: true,
      emit: jest.fn((event: string, payload: any, ack?: (data: any) => void) => {
        if (event === "driver_location_batch") {
          emittedBatches.push(JSON.parse(JSON.stringify(payload)));
          if (typeof ack === "function") ack({ success: true });
        }
      }),
    };
    (getSocket as jest.Mock).mockReturnValue(socket);

    await clearLocationQueue();

    const QUEUE_KEY = "@atmr:location_queue";
    const legacyPosition = {
      latitude: 46.19,
      longitude: 6.14,
      speed: 1.5,
      heading: 90,
      accuracy: 12,
      timestamp: Date.now() - 5000,
      driver_id: 42,
    };
    store.set(QUEUE_KEY, JSON.stringify([legacyPosition]));

    await syncLocationQueue(socket);

    expect(emittedBatches.length).toBe(1);
    const pos = emittedBatches[0].positions[0];
    expect(pos.location_mode).toBe("mission_live");
    expect(typeof pos.recorded_at).toBe("string");
    expect(pos.recorded_at.length).toBeGreaterThan(0);
    expect(typeof pos.sent_at).toBe("string");
    expect(pos.latitude).toBe(46.19);
    expect(pos.longitude).toBe(6.14);
  }, 20000);

  it("getLocationQueue migre les positions legacy à la lecture", async () => {
    const { getLocationQueue } = require("../locationQueue");
    const QUEUE_KEY = "@atmr:location_queue";
    const legacyPosition = {
      latitude: 46.2,
      longitude: 6.1,
      speed: 0,
      heading: 0,
      accuracy: 10,
      timestamp: Date.now(),
      driver_id: 7,
    };
    store.set(QUEUE_KEY, JSON.stringify([legacyPosition]));

    const queue = await getLocationQueue();
    expect(queue[0].location_mode).toBe("mission_live");
    expect(typeof queue[0].recorded_at).toBe("string");
    expect(typeof queue[0].sent_at).toBe("string");
  });

});
