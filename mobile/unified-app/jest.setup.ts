import { jest } from "@jest/globals";
import { Image } from "react-native";

const resolveAssetSource = Image.resolveAssetSource.bind(Image);
Image.resolveAssetSource = (source) => {
  const resolved = resolveAssetSource(source);
  if (resolved?.uri?.trim()) return resolved;
  const id =
    typeof source === "number"
      ? String(source)
      : typeof source === "string"
        ? source
        : "asset";
  return { uri: `https://jest.local/${id}.svg`, width: 56, height: 73, scale: 1 };
};

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
  // eslint-disable-next-line @typescript-eslint/no-require-imports -- jest.mock factory (pas d'import dynamique ES)
  const React = require("react");
  const Icon = () => React.createElement(React.Fragment);
  return {
    __esModule: true,
    Ionicons: Icon,
  };
});

jest.mock("@react-native-async-storage/async-storage", () =>
  jest.requireActual("@react-native-async-storage/async-storage/jest/async-storage-mock")
);

jest.mock("expo-secure-store", () => ({
  __esModule: true,
  getItemAsync: jest.fn(async () => null),
  setItemAsync: jest.fn(async () => undefined),
  deleteItemAsync: jest.fn(async () => undefined),
  WHEN_UNLOCKED: "WHEN_UNLOCKED",
}));

jest.mock("lottie-react-native", () => {
  // eslint-disable-next-line @typescript-eslint/no-require-imports -- jest.mock factory
  const React = require("react");
  const MockLottie = () => React.createElement(React.Fragment);
  return { __esModule: true, default: MockLottie };
});

// Après `jest.preload.cjs` (fetch retiré) : forcer l’adaptateur http Node.
try {
  /* eslint-disable @typescript-eslint/no-require-imports -- bindings Node pour axios en Jest */
  const nodeHttp = require("http");
  const nodeHttps = require("https");
  const axiosLib = require("axios");
  const httpAdapter = require("axios/lib/adapters/http");
  /* eslint-enable @typescript-eslint/no-require-imports */
  axiosLib.defaults.adapter = (config: object) => httpAdapter(config, { http: nodeHttp, https: nodeHttps });
} catch {
  // no-op
}

// Restaurer fetch pour les tests / libs qui s'y attendent (client déjà initialisé).
const _globalWithSavedFetch = globalThis as unknown as {
  __JEST_SAVED_FETCH__?: typeof fetch;
  fetch: typeof fetch;
};
if (typeof _globalWithSavedFetch.__JEST_SAVED_FETCH__ === "function") {
  _globalWithSavedFetch.fetch = _globalWithSavedFetch.__JEST_SAVED_FETCH__;
}

// expo-linking.parse s’appuie sur l’hôte / schéma natifs (Bare) : polyfill Jest.
jest.mock("expo-linking", () => {
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
