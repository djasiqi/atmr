import { createSecureRandomId } from "./secureRandomId";

describe("createSecureRandomId", () => {
  const originalCrypto = globalThis.crypto;
  const originalExpo = (globalThis as { expo?: unknown }).expo;

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

  function clearCryptoAndExpo() {
    Object.defineProperty(globalThis, "crypto", {
      configurable: true,
      value: {},
    });
    Object.defineProperty(globalThis, "expo", {
      configurable: true,
      value: undefined,
    });
  }

  it("utilise crypto.randomUUID et conserve le préfixe", () => {
    const randomUUID = jest.fn(() => "11111111-2222-4333-8444-555555555555");
    Object.defineProperty(globalThis, "crypto", {
      configurable: true,
      value: { randomUUID },
    });
    Object.defineProperty(globalThis, "expo", {
      configurable: true,
      value: { uuidv4: jest.fn(() => "should-not-be-used") },
    });

    const id = createSecureRandomId("atmr-");
    expect(randomUUID).toHaveBeenCalledTimes(1);
    expect(id).toBe("atmr-11111111-2222-4333-8444-555555555555");
  });

  it("D3 — fallback Hermes : expo.uuidv4 si randomUUID absent", () => {
    Object.defineProperty(globalThis, "crypto", {
      configurable: true,
      value: {},
    });
    const uuidv4 = jest.fn(() => "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee");
    Object.defineProperty(globalThis, "expo", {
      configurable: true,
      value: { uuidv4 },
    });

    const id = createSecureRandomId("atmr-");
    expect(uuidv4).toHaveBeenCalledTimes(1);
    expect(id).toBe("atmr-aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee");
  });

  it("produit deux valeurs distinctes (crypto)", () => {
    let n = 0;
    Object.defineProperty(globalThis, "crypto", {
      configurable: true,
      value: {
        randomUUID: () => {
          n += 1;
          return `11111111-2222-4333-8444-55555555555${n}`;
        },
      },
    });
    const first = createSecureRandomId("atmr-");
    const second = createSecureRandomId("atmr-");
    expect(first).not.toBe(second);
    expect(first.startsWith("atmr-")).toBe(true);
    expect(second.startsWith("atmr-")).toBe(true);
  });

  it("D4 — échoue si crypto.randomUUID et expo.uuidv4 absents", () => {
    clearCryptoAndExpo();
    expect(() => createSecureRandomId("atmr-")).toThrow("secure_random_unavailable");
  });

  it("échoue si randomUUID retourne une valeur malformed", () => {
    Object.defineProperty(globalThis, "crypto", {
      configurable: true,
      value: { randomUUID: () => "not-a-uuid" },
    });
    Object.defineProperty(globalThis, "expo", {
      configurable: true,
      value: undefined,
    });
    expect(() => createSecureRandomId("atmr-")).toThrow("secure_random_unavailable");
  });

  it("échoue si expo.uuidv4 retourne null/vide", () => {
    Object.defineProperty(globalThis, "crypto", {
      configurable: true,
      value: {},
    });
    Object.defineProperty(globalThis, "expo", {
      configurable: true,
      value: { uuidv4: () => "" },
    });
    expect(() => createSecureRandomId("atmr-")).toThrow("secure_random_unavailable");
  });
});
