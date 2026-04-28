import { beforeEach, describe, expect, it, jest } from "@jest/globals";
import {
  getItem,
  getParsedOrNull,
  migrateIfNeeded,
  removeItem,
  setItem,
} from "./typedStorage";
import { STORAGE_KEYS } from "./storageKeys";

const storage = new Map<string, string>();

const mockGetItem = jest.fn(async (key: string) => storage.get(key) ?? null);
const mockSetItem = jest.fn(async (key: string, value: string) => {
  storage.set(key, value);
});
const mockRemoveItem = jest.fn(async (key: string) => {
  storage.delete(key);
});

jest.mock("@react-native-async-storage/async-storage", () => ({
  __esModule: true,
  default: {
    getItem: (key: string) => mockGetItem(key),
    setItem: (key: string, value: string) => mockSetItem(key, value),
    removeItem: (key: string) => mockRemoveItem(key),
  },
}));

describe("typedStorage", () => {
  beforeEach(() => {
    storage.clear();
    mockGetItem.mockClear();
    mockSetItem.mockClear();
    mockRemoveItem.mockClear();
  });

  it("returns null when JSON is invalid", async () => {
    storage.set("broken", "{oops");
    const result = await getItem("broken");
    expect(result).toBeNull();
  });

  it("writes and reads typed values", async () => {
    await setItem(STORAGE_KEYS.SESSION_RUNTIME, { mode: "driver", ts: 42 });
    const value = await getItem<{ mode: string; ts: number }>(
      STORAGE_KEYS.SESSION_RUNTIME
    );
    expect(value).toEqual({ mode: "driver", ts: 42 });
    await removeItem(STORAGE_KEYS.SESSION_RUNTIME);
    expect(await getItem(STORAGE_KEYS.SESSION_RUNTIME)).toBeNull();
  });

  it("migrates legacy driver profile key once", async () => {
    storage.set("driver_profile_cache_v1", JSON.stringify({ profile: { id: 7 } }));
    await migrateIfNeeded();

    expect(storage.get(STORAGE_KEYS.DRIVER_PROFILE)).toEqual(
      JSON.stringify({ profile: { id: 7 } })
    );
    expect(storage.has("driver_profile_cache_v1")).toBe(false);
    expect(storage.get(STORAGE_KEYS.STORAGE_MIGRATION_VERSION)).toBe("1");
  });

  it("parses raw JSON safely", () => {
    expect(getParsedOrNull<{ ok: boolean }>(JSON.stringify({ ok: true }))).toEqual({
      ok: true,
    });
    expect(getParsedOrNull("{")).toBeNull();
    expect(getParsedOrNull(null)).toBeNull();
  });
});

