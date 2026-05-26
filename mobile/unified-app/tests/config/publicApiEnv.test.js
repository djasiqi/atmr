const {
  resolveApiBaseUrlFromEnv,
  resolveDriverSocketUrlFromEnv,
  assertProdHttpsEnv,
  resolveProdApiBaseUrlForEas,
  resolveProdDriverSocketUrlForEas,
} = require("../../config/publicApiEnv.cjs");

const ENV_KEYS = [
  "EXPO_PUBLIC_API_BASE_URL",
  "EXPO_PUBLIC_API_URL",
  "EXPO_PUBLIC_DRIVER_SOCKET_URL",
  "EXPO_PUBLIC_SOCKET_URL",
];

describe("publicApiEnv", () => {
  const snapshot = Object.fromEntries(ENV_KEYS.map((key) => [key, process.env[key]]));

  afterEach(() => {
    for (const key of ENV_KEYS) {
      if (snapshot[key] === undefined) delete process.env[key];
      else process.env[key] = snapshot[key];
    }
  });

  function clearApiEnv() {
    for (const key of ENV_KEYS) {
      delete process.env[key];
    }
  }

  it("maps legacy EXPO_PUBLIC_API_URL to /api/v1 base", () => {
    clearApiEnv();
    process.env.EXPO_PUBLIC_API_URL = "https://api.lirie.ch";
    expect(resolveApiBaseUrlFromEnv()).toBe("https://api.lirie.ch/api/v1");
  });

  it("prefers EXPO_PUBLIC_API_BASE_URL over legacy URL", () => {
    clearApiEnv();
    process.env.EXPO_PUBLIC_API_BASE_URL = "https://api.lirie.ch/api/v1";
    process.env.EXPO_PUBLIC_API_URL = "http://192.168.1.1:5000";
    expect(resolveApiBaseUrlFromEnv()).toBe("https://api.lirie.ch/api/v1");
  });

  it("falls back driver socket to EXPO_PUBLIC_SOCKET_URL", () => {
    clearApiEnv();
    process.env.EXPO_PUBLIC_SOCKET_URL = "https://api.lirie.ch";
    expect(resolveDriverSocketUrlFromEnv()).toBe("https://api.lirie.ch");
  });

  it("rejects LAN URLs in prod assert", () => {
    expect(() =>
      assertProdHttpsEnv("EXPO_PUBLIC_API_BASE_URL", "https://192.168.1.146/api/v1")
    ).toThrow(/LAN/);
  });

  it("coerces LAN API URL to prod default on EAS", () => {
    expect(resolveProdApiBaseUrlForEas("http://192.168.1.146:5000/api/v1")).toBe(
      "https://api.lirie.ch/api/v1"
    );
  });

  it("keeps valid prod API URL on EAS", () => {
    expect(resolveProdApiBaseUrlForEas("https://api.lirie.ch/api/v1")).toBe(
      "https://api.lirie.ch/api/v1"
    );
  });

  it("falls back driver socket from legacy EXPO_PUBLIC_SOCKET_URL on EAS", () => {
    clearApiEnv();
    process.env.EXPO_PUBLIC_SOCKET_URL = "https://api.lirie.ch";
    expect(resolveProdDriverSocketUrlForEas("http://192.168.1.146:5000")).toBe("https://api.lirie.ch");
  });
});
