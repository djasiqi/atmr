import { afterEach, beforeEach, describe, expect, it, jest } from "@jest/globals";

// Contrôle de l'environnement avant le chargement du module
let sendIngestEvent: (event: string, payload: Record<string, unknown>) => void;
let resetIngestAdapterForTests: () => void;

const mockFetch = jest.fn() as jest.MockedFunction<typeof fetch>;

jest.mock("react-native", () => ({
  Platform: { OS: "ios" },
}));

jest.mock("expo-constants", () => ({
  default: { isDevice: true, deviceName: "iPhone Test" },
}));

// __DEV__ est défini globalement par Metro/Jest — on le surcharge pour tester les deux chemins
const globalAny = global as Record<string, unknown>;

describe("ingestAdapter", () => {
  beforeEach(() => {
    jest.resetModules();
    mockFetch.mockReset();
    globalAny.__DEV__ = true;
    globalAny.fetch = mockFetch;
    mockFetch.mockResolvedValue(new Response("ok", { status: 200 }));
    delete process.env.EXPO_PUBLIC_DISABLE_INGEST;

    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const mod = require("./ingestAdapter");
    sendIngestEvent = mod.sendIngestEvent;
    resetIngestAdapterForTests = mod.resetIngestAdapterForTests;
    resetIngestAdapterForTests();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it("envoie un POST vers /ingest avec event + ts + payload en __DEV__", async () => {
    sendIngestEvent("driver.gate.unified_evaluated", { source: "test", enabled: true });
    await Promise.resolve();

    expect(mockFetch).toHaveBeenCalledTimes(1);
    const [url, init] = mockFetch.mock.calls[0] as [string, RequestInit];
    expect(url).toContain("/ingest");
    expect(url).toContain("7242");
    const body = JSON.parse(init.body as string);
    expect(body.event).toBe("driver.gate.unified_evaluated");
    expect(body.ts).toBeDefined();
    expect(body.source).toBe("test");
    expect(body.enabled).toBe(true);
  });

  it("ne fait rien quand EXPO_PUBLIC_DISABLE_INGEST=1", async () => {
    jest.resetModules();
    process.env.EXPO_PUBLIC_DISABLE_INGEST = "1";

    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const mod = require("./ingestAdapter");
    mod.resetIngestAdapterForTests();
    mod.sendIngestEvent("any.event", {});
    await Promise.resolve();

    expect(mockFetch).not.toHaveBeenCalled();
    delete process.env.EXPO_PUBLIC_DISABLE_INGEST;
  });

  it("suspend automatiquement après 3 échecs consécutifs", async () => {
    mockFetch.mockRejectedValue(new Error("network error"));

    sendIngestEvent("ev1", {});
    sendIngestEvent("ev2", {});
    sendIngestEvent("ev3", {});
    for (let i = 0; i < 20; i += 1) {
      await Promise.resolve();
    }

    mockFetch.mockReset();
    sendIngestEvent("ev4", {});
    await Promise.resolve();

    // Après 3 échecs, le 4e appel doit être silencieusement ignoré
    expect(mockFetch).not.toHaveBeenCalled();
  });

  it("reset rétablit l'état initial (suspended=false, failures=0)", async () => {
    mockFetch.mockRejectedValue(new Error("fail"));
    sendIngestEvent("e1", {});
    sendIngestEvent("e2", {});
    sendIngestEvent("e3", {});
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();

    resetIngestAdapterForTests();
    mockFetch.mockReset();
    mockFetch.mockResolvedValue(new Response("ok", { status: 200 }));

    sendIngestEvent("e4", {});
    await Promise.resolve();

    expect(mockFetch).toHaveBeenCalledTimes(1);
  });

  it("utilise 10.0.2.2 sur émulateur Android", async () => {
    jest.resetModules();
    jest.doMock("react-native", () => ({ Platform: { OS: "android" } }));
    jest.doMock("expo-constants", () => ({
      default: { isDevice: false, deviceName: "Android Emulator" },
    }));

    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const mod = require("./ingestAdapter");
    mod.resetIngestAdapterForTests();
    mod.sendIngestEvent("test.event", {});
    await Promise.resolve();

    const [url] = mockFetch.mock.calls[0] as [string, RequestInit];
    expect(url).toContain("10.0.2.2");
  });
});
