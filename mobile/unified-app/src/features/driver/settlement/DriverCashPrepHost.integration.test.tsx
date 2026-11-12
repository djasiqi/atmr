import { createElement } from "react";
import TestRenderer, { act } from "react-test-renderer";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import http from "http";
import { Text } from "react-native";
import { DriverCashPrepHost } from "./DriverCashPrepHost";

function request(
  method: string,
  url: string,
  token: string | null,
  body?: Record<string, unknown>
): Promise<{ data: Record<string, unknown> }> {
  return new Promise((resolve, reject) => {
    const target = new URL(url);
    const payload = body ? JSON.stringify(body) : undefined;
    const call = http.request(
      {
        hostname: target.hostname,
        port: target.port,
        path: `${target.pathname}${target.search}`,
        method,
        headers: {
          "Content-Type": "application/json",
          "X-Client-Platform": "ios",
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
          ...(payload ? { "Content-Length": Buffer.byteLength(payload) } : {}),
        },
      },
      (response) => {
        const chunks: Buffer[] = [];
        response.on("data", (chunk) => chunks.push(chunk));
        response.on("end", () => {
          const raw = Buffer.concat(chunks).toString("utf8");
          const parsed = raw ? JSON.parse(raw) : {};
          const status = response.statusCode || 500;
          if (status >= 400) {
            const error = new Error(`http ${status}`) as Error & {
              response?: { status: number };
            };
            error.response = { status };
            reject(error);
            return;
          }
          resolve({ data: parsed });
        });
      }
    );
    call.on("error", reject);
    if (payload) call.write(payload);
    call.end();
  });
}

const base = process.env.CASH_RECIPE_BASE;
const describeLive = base ? describe : describe.skip;
jest.setTimeout(30000);

async function login(): Promise<string> {
  const response = await request("POST", `${base}/api/v1/auth/login`, null, {
    email: "ch-a37-1@lirie-e2e.invalid",
    password: ["local", "test", "fixture", "p0gg"].join("-"),
  });
  return String(response.data.access_token);
}

function clientFor(token: string, dropDeclare: { current: boolean }) {
  return {
    api: {
      get: (url: string) => request("GET", `${base}/api/v1${url}`, token),
      post: async (url: string, body?: Record<string, unknown>) => {
        const response = await request("POST", `${base}/api/v1${url}`, token, body);
        if (dropDeclare.current && body?.action === "declare_cash") {
          dropDeclare.current = false;
          throw new Error("réponse perdue");
        }
        return response;
      },
    },
  };
}

jest.mock("../../../core/api/client", () => ({
  apiClient: {
    get: (...args: unknown[]) => mocked.api.get(...(args as [string])),
    post: (...args: unknown[]) =>
      mocked.api.post(...(args as [string, Record<string, unknown>])),
  },
}));

const mocked = {
  api: {
    get: (_url: string) => Promise.reject(new Error("non prêt")),
    post: (_url: string, _body?: Record<string, unknown>) =>
      Promise.reject(new Error("non prêt")),
  },
};

function textOf(tree: TestRenderer.ReactTestRenderer): string {
  return tree.root
    .findAllByType(Text)
    .map((node) => {
      const children = node.props.children;
      return Array.isArray(children) ? children.join("") : String(children ?? "");
    })
    .join(" ");
}

async function show(missionId: number) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  let tree!: TestRenderer.ReactTestRenderer;
  await act(async () => {
    tree = TestRenderer.create(
      createElement(
        QueryClientProvider,
        { client: queryClient },
        createElement(DriverCashPrepHost, { missionId, onBack: jest.fn() })
      )
    );
  });
  for (let attempt = 0; attempt < 20; attempt += 1) {
    if (textOf(tree).includes("45.00")) return tree;
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 150));
    });
  }
  throw new Error(textOf(tree) || "écran vide");
}

async function typeAmount(tree: TestRenderer.ReactTestRenderer, value: string) {
  await act(async () => {
    tree.root.findByProps({ accessibilityLabel: "Montant remis par le patient" }).props.onChangeText(value);
  });
}

async function press(tree: TestRenderer.ReactTestRenderer, label: string) {
  await act(async () => {
    tree.root.findByProps({ accessibilityLabel: label }).props.onPress();
    await new Promise((resolve) => setTimeout(resolve, 400));
  });
}

describeLive("écran espèces raccordé à la recette", () => {
  let token = "";

  beforeAll(async () => {
    token = await login();
    const drop = { current: false };
    mocked.api = clientFor(token, drop).api;
  });

  it("encaisse le montant exact sans champ modifiable", async () => {
    const tree = await show(9);
    expect(textOf(tree)).toContain("verrouillé");
    expect(tree.root.findAllByType(require("react-native").TextInput)).toHaveLength(0);
    if (!textOf(tree).includes("Réglé")) {
      await typeAmount(tree, "45.00");
      await press(tree, "Confirmer la réception");
    }
    expect(textOf(tree)).toContain("Réglé");
  });

  it("confirme la monnaie rendue", async () => {
    const tree = await show(10);
    if (!textOf(tree).includes("Réglé")) {
      await press(tree, "Confirmer la monnaie rendue");
    }
    expect(textOf(tree)).toContain("Réglé");
  });

  it("restitue une somme insuffisante", async () => {
    const tree = await show(11);
    if (!textOf(tree).includes("Espèces restituées")) {
      await press(tree, "Restituer la somme insuffisante");
    }
    expect(textOf(tree)).toContain("Espèces restituées");
  });

  it("conserve administrativement sans quittance", async () => {
    const tree = await show(12);
    if (!textOf(tree).includes("Conservation administrative")) {
      await press(tree, "Conserver administrativement");
    }
    expect(textOf(tree)).toContain("Conservation administrative");
    expect(textOf(tree)).not.toContain("Spécimen");
  });

  it("ne renvoie pas une réception dont la réponse a été perdue", async () => {
    const drop = { current: true };
    const bridge = clientFor(token, drop);
    mocked.api = bridge.api;
    const lostId = Number(process.env.CASH_LOST_BOOKING || "13");
    const tree = await show(lostId);
    await typeAmount(tree, "45.00");
      await press(tree, "Confirmer la réception");
    expect(textOf(tree)).toContain("Aucune seconde réception automatique");
    await press(tree, "Relire le dossier");
    expect(textOf(tree)).toContain("Réglé");
    const again = await request(
      "GET",
      `${base}/api/v1/driver/me/bookings/${lostId}/cash-collection`,
      token
    );
    const declarations = (again.data.history as { action: string }[]).filter(
      (event: { action: string }) => event.action === "declare_cash"
    );
    expect(declarations).toHaveLength(1);
  });
});
