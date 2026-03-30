import AsyncStorage from "@react-native-async-storage/async-storage";
import {
  DRIVER_PROFILE_CACHE_SCHEMA_VERSION,
  DRIVER_PROFILE_CACHE_STORAGE_KEY,
  DRIVER_PROFILE_CACHE_TTL_MS,
  readDriverProfileCache,
  purgeDriverProfileCache,
  invalidateDriverProfileCacheIfUserMismatch,
  writeDriverProfileCache,
} from "../driverProfileCache";
import { asyncStorage, secureStorage } from "../storage";
import type { Driver } from "../api";

jest.mock("expo-secure-store", () => ({
  setItemAsync: jest.fn(),
  getItemAsync: jest.fn(),
  deleteItemAsync: jest.fn(),
}));

jest.mock("@react-native-async-storage/async-storage", () => ({
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn().mockResolvedValue(undefined),
  multiRemove: jest.fn().mockResolvedValue(undefined),
}));

jest.mock("@/src/config/telemetry", () => ({
  sendIngestEvent: jest.fn(),
}));

function minimalDriver(overrides: Partial<Driver> = {}): Driver {
  return {
    id: 7,
    user_id: 99,
    username: "d",
    first_name: "A",
    last_name: "B",
    phone: "0",
    photo: "",
    company_id: 1,
    company_name: "C",
    is_active: true,
    is_available: true,
    vehicle_assigned: "",
    brand: "",
    license_plate: "",
    latitude: null,
    longitude: null,
    user: {
      id: 99,
      username: "d",
      email: "d@test",
      role: "driver",
      public_id: "pub-99",
    },
    company: { id: 1, name: "C" },
    ...overrides,
  };
}

describe("driverProfileCache", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    (AsyncStorage.getItem as jest.Mock).mockResolvedValue(null);
    jest.spyOn(asyncStorage, "getDriverId").mockResolvedValue(7);
    jest.spyOn(secureStorage, "getUserPublicId").mockResolvedValue("pub-99");
  });

  it("readDriverProfileCache: miss quand absent", async () => {
    const r = await readDriverProfileCache({ allowStale: false });
    expect(r.status).toBe("miss");
    expect(r.profile).toBeUndefined();
  });

  it("readDriverProfileCache: hit quand TTL valide et identité ok", async () => {
    const profile = minimalDriver();
    const envelope = {
      schema_version: DRIVER_PROFILE_CACHE_SCHEMA_VERSION,
      profile,
      driver_id: profile.id,
      company_id: profile.company_id,
      cached_at_ms: Date.now() - 60_000,
    };
    (AsyncStorage.getItem as jest.Mock).mockResolvedValue(JSON.stringify(envelope));

    const r = await readDriverProfileCache({ allowStale: false });
    expect(r.status).toBe("hit");
    expect(r.profile?.id).toBe(7);
  });

  it("readDriverProfileCache: expired sans allowStale — pas de profil", async () => {
    const profile = minimalDriver();
    const envelope = {
      schema_version: DRIVER_PROFILE_CACHE_SCHEMA_VERSION,
      profile,
      driver_id: profile.id,
      company_id: profile.company_id,
      cached_at_ms: Date.now() - DRIVER_PROFILE_CACHE_TTL_MS - 1000,
    };
    (AsyncStorage.getItem as jest.Mock).mockResolvedValue(JSON.stringify(envelope));

    const r = await readDriverProfileCache({ allowStale: false });
    expect(r.status).toBe("expired");
    expect(r.profile).toBeUndefined();
  });

  it("readDriverProfileCache: expired avec allowStale — retourne profil", async () => {
    const profile = minimalDriver();
    const envelope = {
      schema_version: DRIVER_PROFILE_CACHE_SCHEMA_VERSION,
      profile,
      driver_id: profile.id,
      company_id: profile.company_id,
      cached_at_ms: Date.now() - DRIVER_PROFILE_CACHE_TTL_MS - 1000,
    };
    (AsyncStorage.getItem as jest.Mock).mockResolvedValue(JSON.stringify(envelope));

    const r = await readDriverProfileCache({ allowStale: true });
    expect(r.status).toBe("expired");
    expect(r.profile?.id).toBe(7);
  });

  it("readDriverProfileCache: schema_mismatch", async () => {
    const profile = minimalDriver();
    const envelope = {
      schema_version: 999,
      profile,
      driver_id: profile.id,
      company_id: profile.company_id,
      cached_at_ms: Date.now(),
    };
    (AsyncStorage.getItem as jest.Mock).mockResolvedValue(JSON.stringify(envelope));

    const r = await readDriverProfileCache({ allowStale: false });
    expect(r.status).toBe("schema_mismatch");
  });

  it("readDriverProfileCache: driver_mismatch si driver_id stocké différent", async () => {
    jest.spyOn(asyncStorage, "getDriverId").mockResolvedValue(999);
    const profile = minimalDriver({ id: 7 });
    const envelope = {
      schema_version: DRIVER_PROFILE_CACHE_SCHEMA_VERSION,
      profile,
      driver_id: profile.id,
      company_id: profile.company_id,
      cached_at_ms: Date.now(),
    };
    (AsyncStorage.getItem as jest.Mock).mockResolvedValue(JSON.stringify(envelope));

    const r = await readDriverProfileCache({ allowStale: false });
    expect(r.status).toBe("driver_mismatch");
  });

  it("purgeDriverProfileCache supprime la clé", async () => {
    await purgeDriverProfileCache();
    expect(AsyncStorage.removeItem).toHaveBeenCalledWith(
      DRIVER_PROFILE_CACHE_STORAGE_KEY
    );
  });

  it("invalidateDriverProfileCacheIfUserMismatch purge si user_id différent", async () => {
    const profile = minimalDriver({ user_id: 99 });
    const envelope = {
      schema_version: DRIVER_PROFILE_CACHE_SCHEMA_VERSION,
      profile,
      driver_id: profile.id,
      company_id: profile.company_id,
      cached_at_ms: Date.now(),
    };
    (AsyncStorage.getItem as jest.Mock).mockResolvedValue(JSON.stringify(envelope));

    await invalidateDriverProfileCacheIfUserMismatch(100);
    expect(AsyncStorage.removeItem).toHaveBeenCalledWith(
      DRIVER_PROFILE_CACHE_STORAGE_KEY
    );
  });

  it("writeDriverProfileCache puis lecture — cohérence (mutation locale)", async () => {
    const stored: Record<string, string> = {};
    (AsyncStorage.setItem as jest.Mock).mockImplementation((k: string, v: string) => {
      stored[k] = v;
      return Promise.resolve();
    });
    (AsyncStorage.getItem as jest.Mock).mockImplementation((k: string) =>
      Promise.resolve(stored[k] ?? null)
    );

    const profile = minimalDriver({ first_name: "Updated" });
    await writeDriverProfileCache(profile);

    const r = await readDriverProfileCache({ allowStale: false });
    expect(r.status).toBe("hit");
    expect(r.profile?.first_name).toBe("Updated");
  });
});
