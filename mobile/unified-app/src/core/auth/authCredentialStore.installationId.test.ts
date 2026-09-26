import * as SecureStore from "expo-secure-store";
import {
  createAndPersistInstallationId,
  readInstallationId,
} from "./authCredentialStore";

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

const store = (
  SecureStore as typeof SecureStore & { __store: Map<string, string> }
).__store;

describe("createAndPersistInstallationId — Device Identity D1–D7", () => {
  const originalCrypto = globalThis.crypto;
  const originalExpo = (globalThis as { expo?: unknown }).expo;

  beforeEach(() => {
    store.clear();
    jest.clearAllMocks();
  });

  afterEach(() => {
    Object.defineProperty(globalThis, "crypto", {
      configurable: true,
      value: originalCrypto,
    });
    Object.defineProperty(globalThis, "expo", {
      configurable: true,
      value: originalExpo,
    });
  });

  function mockCryptoUuid(uuid: string) {
    Object.defineProperty(globalThis, "crypto", {
      configurable: true,
      value: { randomUUID: jest.fn(() => uuid) },
    });
  }

  function mockExpoOnly(uuid: string) {
    Object.defineProperty(globalThis, "crypto", {
      configurable: true,
      value: {},
    });
    Object.defineProperty(globalThis, "expo", {
      configurable: true,
      value: { uuidv4: jest.fn(() => uuid) },
    });
  }

  function clearRng() {
    Object.defineProperty(globalThis, "crypto", {
      configurable: true,
      value: {},
    });
    Object.defineProperty(globalThis, "expo", {
      configurable: true,
      value: undefined,
    });
  }

  it("D1 — installation_id existant → exact même ID, aucune génération", async () => {
    store.set("atmr.auth.installation_id", "atmr-legacy-existing");
    const randomUUID = jest.fn(() => "11111111-2222-4333-8444-555555555555");
    Object.defineProperty(globalThis, "crypto", {
      configurable: true,
      value: { randomUUID },
    });
    const first = await createAndPersistInstallationId();
    const second = await createAndPersistInstallationId();
    expect(first).toEqual({ status: "found", value: "atmr-legacy-existing" });
    expect(second).toEqual({ status: "found", value: "atmr-legacy-existing" });
    expect(randomUUID).not.toHaveBeenCalled();
  });

  it("D2 — missing + randomUUID → génération + persist + read-back", async () => {
    mockCryptoUuid("aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee");
    const created = await createAndPersistInstallationId();
    expect(created).toEqual({
      status: "found",
      value: "atmr-aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee",
    });
    await expect(readInstallationId()).resolves.toEqual(created);
  });

  it("D3 — randomUUID absent + expo.uuidv4 → ID sécurisé persisté", async () => {
    mockExpoOnly("ffffffff-eeee-4ddd-8ccc-bbbbbbbbbbbb");
    const created = await createAndPersistInstallationId();
    expect(created).toEqual({
      status: "found",
      value: "atmr-ffffffff-eeee-4ddd-8ccc-bbbbbbbbbbbb",
    });
    await expect(readInstallationId()).resolves.toEqual(created);
  });

  it("D4 — aucun CSPRNG → secure_random_unavailable fail-closed", async () => {
    clearRng();
    const result = await createAndPersistInstallationId();
    expect(result.status).toBe("temporarily_unavailable");
    if (result.status === "temporarily_unavailable") {
      expect(result.cause).toBe("secure_random_unavailable");
    }
    await expect(readInstallationId()).resolves.toEqual({ status: "missing" });
  });

  it("D5 — SecureStore write fail → temporarily_unavailable (pas d'ID)", async () => {
    mockCryptoUuid("aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee");
    (SecureStore.setItemAsync as jest.Mock).mockRejectedValueOnce(
      new Error("keystore_locked")
    );
    const result = await createAndPersistInstallationId();
    expect(result.status).toBe("temporarily_unavailable");
    await expect(readInstallationId()).resolves.toEqual({ status: "missing" });
  });

  it("D6 — write ok mais read-back mismatch → DEVICE path unavailable", async () => {
    mockCryptoUuid("aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee");
    (SecureStore.setItemAsync as jest.Mock).mockImplementationOnce(
      async (key: string, value: string) => {
        store.set(key, `${value}-TAMPERED`);
      }
    );
    const result = await createAndPersistInstallationId();
    expect(result.status).toBe("temporarily_unavailable");
    if (result.status === "temporarily_unavailable") {
      expect(result.cause).toMatch(/read_back_mismatch|device_identity_storage_unavailable/);
    }
  });

  it("D7 — restart simulé : cache SecureStore conserve le même ID", async () => {
    mockCryptoUuid("aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee");
    const created = await createAndPersistInstallationId();
    expect(created.status).toBe("found");
    // Simule restart : pas de régénération, relecture store
    clearRng();
    const restored = await createAndPersistInstallationId();
    expect(restored).toEqual(created);
  });
});
