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
    __store: store,
  };
});

const store = (
  SecureStore as typeof SecureStore & { __store: Map<string, string> }
).__store;

describe("createAndPersistInstallationId", () => {
  const originalCrypto = globalThis.crypto;

  beforeEach(() => {
    store.clear();
    jest.clearAllMocks();
  });

  afterEach(() => {
    Object.defineProperty(globalThis, "crypto", {
      configurable: true,
      value: originalCrypto,
    });
  });

  it("conserve un ID déjà persisté", async () => {
    store.set("atmr.auth.installation_id", "atmr-legacy-existing");
    const first = await createAndPersistInstallationId();
    const second = await createAndPersistInstallationId();
    expect(first).toEqual({ status: "found", value: "atmr-legacy-existing" });
    expect(second).toEqual({ status: "found", value: "atmr-legacy-existing" });
  });

  it("génère un ID CSPRNG et le relit", async () => {
    const randomUUID = jest.fn(() => "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee");
    Object.defineProperty(globalThis, "crypto", {
      configurable: true,
      value: { randomUUID },
    });

    const created = await createAndPersistInstallationId();
    expect(randomUUID).toHaveBeenCalled();
    expect(created).toEqual({
      status: "found",
      value: "atmr-aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee",
    });
    await expect(readInstallationId()).resolves.toEqual(created);
  });

  it("échoue sans régénérer si le CSPRNG est absent", async () => {
    Object.defineProperty(globalThis, "crypto", {
      configurable: true,
      value: {},
    });
    const result = await createAndPersistInstallationId();
    expect(result.status).toBe("temporarily_unavailable");
    if (result.status === "temporarily_unavailable") {
      expect(result.cause).toBe("secure_random_unavailable");
    }
    await expect(readInstallationId()).resolves.toEqual({ status: "missing" });
  });
});
