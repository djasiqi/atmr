import AsyncStorage from "@react-native-async-storage/async-storage";
import {
  enqueueLocation,
  enqueueLocationBatch,
  getLocationQueue,
  clearLocationQueue,
  type QueuedLocation,
} from "../locationQueue";

jest.mock("expo-crypto", () => ({
  randomUUID: jest.fn(() => "aaaaaaaa-bbbb-4ccc-dddd-eeeeeeeeeeee"),
}));

jest.mock("@react-native-async-storage/async-storage", () => ({
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn(),
}));

const store = new Map<string, string>();

beforeEach(() => {
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
});

function mkLoc(
  i: number,
  mode: QueuedLocation["location_mode"] = "mission_live"
): QueuedLocation {
  return {
    latitude: 46.2 + i * 0.0001,
    longitude: 6.1 + i * 0.0001,
    speed: 10,
    heading: 120,
    accuracy: 8,
    timestamp: Date.now() + i,
    driver_id: 1,
    location_mode: mode,
  };
}

describe("locationQueue v2.1 retention", () => {
  it("enqueueLocationBatch assigns location_event_id like enqueueLocation", async () => {
    await clearLocationQueue();
    const ts = Date.now();
    await enqueueLocationBatch([
      {
        latitude: 46.2,
        longitude: 6.1,
        speed: 1,
        heading: 0,
        accuracy: 10,
        timestamp: ts,
        driver_id: 99,
        location_mode: "mission_live",
      },
    ]);
    const queue = await getLocationQueue();
    expect(queue.length).toBeGreaterThan(0);
    const last = queue[queue.length - 1];
    expect(last.location_event_id).toBe("aaaaaaaa-bbbb-4ccc-dddd-eeeeeeeeeeee");
  });

  it("keeps only latest availability_presence per driver", async () => {
    await clearLocationQueue();
    const first = mkLoc(1, "availability_presence");
    const second = mkLoc(2, "availability_presence");
    await enqueueLocation(first);
    await enqueueLocation(second);
    const queue = await getLocationQueue();
    const avail = queue.filter((q) => q.location_mode === "availability_presence");
    expect(avail).toHaveLength(1);
    expect(avail[0].timestamp).toBe(second.timestamp);
  });

  it("caps mission_live buffered points to 20", async () => {
    await clearLocationQueue();
    for (let i = 0; i < 25; i++) {
      await enqueueLocation(mkLoc(i, "mission_live"));
    }
    const queue = await getLocationQueue();
    const mission = queue.filter((q) => q.location_mode === "mission_live");
    expect(mission.length).toBeLessThanOrEqual(20);
  });
});
