import { createSecureRandomId } from "./secureRandomId";

describe("createSecureRandomId", () => {
  const originalCrypto = globalThis.crypto;

  afterEach(() => {
    Object.defineProperty(globalThis, "crypto", {
      configurable: true,
      value: originalCrypto,
    });
  });

  it("utilise crypto.randomUUID et conserve le préfixe", () => {
    const randomUUID = jest.fn(() => "11111111-2222-4333-8444-555555555555");
    Object.defineProperty(globalThis, "crypto", {
      configurable: true,
      value: { randomUUID },
    });

    const id = createSecureRandomId("atmr-");
    expect(randomUUID).toHaveBeenCalledTimes(1);
    expect(id).toBe("atmr-11111111-2222-4333-8444-555555555555");
  });

  it("produit deux valeurs distinctes", () => {
    const first = createSecureRandomId("atmr-");
    const second = createSecureRandomId("atmr-");
    expect(first).not.toBe(second);
    expect(first.startsWith("atmr-")).toBe(true);
    expect(second.startsWith("atmr-")).toBe(true);
  });

  it("échoue explicitement si le CSPRNG est absent", () => {
    Object.defineProperty(globalThis, "crypto", {
      configurable: true,
      value: {},
    });
    expect(() => createSecureRandomId("atmr-")).toThrow("secure_random_unavailable");
  });
});
