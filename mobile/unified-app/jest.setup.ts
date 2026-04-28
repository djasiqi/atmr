import { jest } from "@jest/globals";

// `expo-constants` : évite l’accès natif (p.ex. EXDevLauncher) lors d’import indirects.
jest.mock("expo-constants", () => ({
  __esModule: true,
  default: {
    appOwnership: "standalone",
    executionEnvironment: "standalone",
    experienceUrl: "https://expo.test",
    // Champs requis par expo-linking (Schemes / parse) en environnement Jest
    expoConfig: {
      name: "unified-test",
      slug: "unified",
      scheme: "lirie",
      extra: { apiBaseUrl: "https://api.test/api/v1" },
    },
  },
}));

jest.mock("expo-font", () => ({
  __esModule: true,
  loadAsync: () => Promise.resolve(),
  isLoaded: () => true,
}));

jest.mock("@expo/vector-icons", () => {
  // eslint-disable-next-line @typescript-eslint/no-require-imports, @typescript-eslint/no-var-requires
  const React = require("react");
  // eslint-disable-next-line @typescript-eslint/explicit-function-return-type
  const Icon = () => React.createElement(React.Fragment);
  return {
    __esModule: true,
    Ionicons: Icon,
  };
});

jest.mock("@react-native-async-storage/async-storage", () =>
  jest.requireActual("@react-native-async-storage/async-storage/jest/async-storage-mock")
);

// Après `jest.preload.cjs` (fetch retiré) : forcer l’adaptateur http Node.
try {
  // eslint-disable-next-line @typescript-eslint/no-require-imports, @typescript-eslint/no-var-requires
  const nodeHttp = require("http");
  // eslint-disable-next-line @typescript-eslint/no-require-imports, @typescript-eslint/no-var-requires
  const nodeHttps = require("https");
  // eslint-disable-next-line @typescript-eslint/no-require-imports, @typescript-eslint/no-var-requires
  const axiosLib = require("axios");
  // eslint-disable-next-line @typescript-eslint/no-require-imports, @typescript-eslint/no-var-requires
  const httpAdapter = require("axios/lib/adapters/http");
  axiosLib.defaults.adapter = (config: object) => httpAdapter(config, { http: nodeHttp, https: nodeHttps });
} catch {
  // no-op
}

// Restaurer fetch pour les tests / libs qui s’y attendent (client déjà initialisé).
if (typeof (globalThis as { __JEST_SAVED_FETCH__?: typeof fetch }).__JEST_SAVED_FETCH__ === "function") {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (globalThis as any).fetch = (globalThis as { __JEST_SAVED_FETCH__: typeof fetch }).__JEST_SAVED_FETCH__;
}

// expo-linking.parse s’appuie sur l’hôte / schéma natifs (Bare) : polyfill Jest.
jest.mock("expo-linking", () => {
  // eslint-disable-next-line @typescript-eslint/explicit-function-return-type
  function parseLink(url: string) {
    if (typeof url !== "string" || !url) {
      return { scheme: null, hostname: "", path: "", queryParams: {} as Record<string, string> };
    }
    if (url.startsWith("lirie://")) {
      const rest = url.slice("lirie://".length);
      const qIdx = rest.indexOf("?");
      const hostPath = (qIdx === -1 ? rest : rest.slice(0, qIdx)) || "";
      const host = hostPath.split("/").filter(Boolean)[0] ?? "";
      const path = `/${hostPath}`.replace(/\/{2,}/g, "/");
      const query: Record<string, string> = {};
      if (qIdx !== -1) {
        new URLSearchParams(rest.slice(qIdx + 1)).forEach((v, k) => {
          query[k] = v;
        });
      }
      return { scheme: "lirie", hostname: host, path, queryParams: query };
    }
    try {
      const u = new URL(url);
      const queryParams: Record<string, string> = {};
      u.searchParams.forEach((v, k) => {
        queryParams[k] = v;
      });
      return { scheme: u.protocol, hostname: u.hostname, path: u.pathname, queryParams };
    } catch {
      return { scheme: null, hostname: "", path: "", queryParams: {} };
    }
  }
  return {
    __esModule: true,
    default: { parse: parseLink, addEventListener: () => ({ remove: () => undefined }), getInitialURL: () => null },
    parse: parseLink,
    createURL: (path: string) => `lirie://${path.replace(/^\//, "")}`,
  };
});
