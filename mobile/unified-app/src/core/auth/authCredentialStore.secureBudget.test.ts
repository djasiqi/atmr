import AsyncStorage from "@react-native-async-storage/async-storage";
import * as SecureStore from "expo-secure-store";
import {
  SECURE_STORE_VALUE_BUDGET_BYTES,
  secureStoreUtf8ByteLength,
  writeRefreshToken,
  writeSessionEnvelope,
  type SessionEnvelope,
} from "./authCredentialStore";
import { OFFLINE_UI_SNAPSHOT_KEY, readOfflineUiSnapshot } from "./offlineUiSnapshotStore";

jest.mock("expo-secure-store", () => {
  const store = new Map<string, string>();
  return {
    AFTER_FIRST_UNLOCK: 1,
    getItemAsync: jest.fn(async (key: string) => store.get(key) ?? null),
    setItemAsync: jest.fn(async (key: string, value: string) => {
      store.set(key, value);
    }),
    deleteItemAsync: jest.fn(async (key: string) => {
      store.delete(key);
    }),
    __store: store,
  };
});

jest.mock("@react-native-async-storage/async-storage", () => {
  const store = new Map<string, string>();
  return {
    getItem: jest.fn(async (key: string) => store.get(key) ?? null),
    setItem: jest.fn(async (key: string, value: string) => {
      store.set(key, value);
    }),
    removeItem: jest.fn(async (key: string) => {
      store.delete(key);
    }),
    __store: store,
  };
});

const secureStore = (
  SecureStore as typeof SecureStore & { __store: Map<string, string> }
).__store;

function envelope(overrides: Partial<SessionEnvelope> = {}): SessionEnvelope {
  return {
    schema_version: 1,
    session_id: "sess-1",
    device_installation_id: "install-1",
    user_public_id: "user-1",
    driver_id: null,
    role: "company",
    active_context_id: "company:1",
    refresh_generation: 2,
    last_authenticated_at: "2026-09-27T20:00:00.000Z",
    ...overrides,
  };
}

describe("budget SecureStore", () => {
  beforeEach(() => {
    secureStore.clear();
    jest.clearAllMocks();
  });

  it("refuse une valeur au-dessus de 1500 octets", async () => {
    const huge = "x".repeat(SECURE_STORE_VALUE_BUDGET_BYTES + 50);
    const result = await writeRefreshToken(huge);
    expect(result.status).toBe("failed");
    if (result.status === "failed") {
      expect(result.cause.startsWith("secure_store_budget_exceeded:")).toBe(true);
    }
    expect(SecureStore.setItemAsync).not.toHaveBeenCalled();
  });

  it("sort le bootstrap du SecureStore et le garde sous le budget", async () => {
    const bootstrap = {
      user: { public_id: "user-1", role: "company" },
      available_contexts: [],
      feature_flags: {},
      padding: "p".repeat(4000),
    };
    const result = await writeSessionEnvelope(
      envelope({
        cached_bootstrap: bootstrap as SessionEnvelope["cached_bootstrap"],
        cached_active_context: {
          context_type: "company",
          context_id: "company:1",
          label: "Entreprise",
          permissions: [],
          is_default: true,
          company_id: 1,
        } as SessionEnvelope["cached_active_context"],
      })
    );
    expect(result.status).toBe("ok");
    const stored = [...secureStore.values()];
    expect(stored.length).toBeGreaterThan(0);
    for (const value of stored) {
      expect(secureStoreUtf8ByteLength(value)).toBeLessThanOrEqual(SECURE_STORE_VALUE_BUDGET_BYTES);
      expect(value.includes("padding")).toBe(false);
    }
    const snapshot = await readOfflineUiSnapshot();
    expect(snapshot?.session_id).toBe("sess-1");
    expect(snapshot?.bootstrap).toEqual(bootstrap);
    expect(AsyncStorage.setItem).toHaveBeenCalledWith(
      OFFLINE_UI_SNAPSHOT_KEY,
      expect.stringContaining("sess-1")
    );
  });
});
