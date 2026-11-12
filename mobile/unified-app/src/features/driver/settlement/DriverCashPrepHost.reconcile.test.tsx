import { createElement } from "react";
import TestRenderer, { act } from "react-test-renderer";
import { QueryClient, QueryClientProvider, focusManager, notifyManager, onlineManager } from "@tanstack/react-query";
import { timeoutManager } from "@tanstack/query-core";
import { Text } from "react-native";
import { apiClient } from "../../../core/api/client";
import { DriverCashPrepHost } from "./DriverCashPrepHost";
import { clearSettlementAttempts } from "./settlementReconciliation";

jest.mock("../../../core/api/client", () => ({
  apiClient: {
    get: jest.fn(),
    post: jest.fn(),
  },
}));

const prepared = {
  booking_id: 3,
  company_name: "Recette",
  debtor_label: "Patient perdu",
  amount_due_chf: "45.00",
  currency: "CHF",
  state: "prepared",
  banner: "PRÉPARATION — ENCAISSEMENT NON ACTIVÉ",
  production_activation: false,
};

const received = {
  ...prepared,
  state: "received",
  tendered_chf: "45.00",
  change_chf: "0.00",
  net_chf: "45.00",
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

function client(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: { retry: false, staleTime: 30_000, gcTime: Infinity, refetchOnMount: "ifStale" },
      mutations: { retry: 1 },
    },
  });
}

notifyManager.setScheduler((callback) => {
  callback();
});

timeoutManager.setTimeoutProvider({
  setTimeout: (callback, delay) => {
    const id = setTimeout(callback, delay);
    id.unref?.();
    return id;
  },
  clearTimeout: (id) => clearTimeout(id as ReturnType<typeof setTimeout>),
  setInterval: (callback, delay) => {
    const id = setInterval(callback, delay);
    id.unref?.();
    return id;
  },
  clearInterval: (id) => clearInterval(id as ReturnType<typeof setInterval>),
});

const mounted: { tree?: TestRenderer.ReactTestRenderer; queryClient?: QueryClient } = {};

async function mount(queryClient: QueryClient, missionId = 3) {
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
  mounted.tree = tree;
  mounted.queryClient = queryClient;
  return tree;
}

function closeScreen() {
  mounted.tree?.unmount();
  mounted.tree = undefined;
}

async function until(tree: TestRenderer.ReactTestRenderer, needle: string) {
  for (let attempt = 0; attempt < 30; attempt += 1) {
    if (textOf(tree).includes(needle)) return;
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 20));
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
  });
}

describe("réconciliation espèces après réponse perdue", () => {
  const get = apiClient.get as jest.Mock;
  const post = apiClient.post as jest.Mock;

  beforeEach(() => {
    clearSettlementAttempts();
    get.mockReset();
    post.mockReset();
  });

  afterEach(async () => {
    await act(async () => {
      closeScreen();
    });
    mounted.queryClient?.unmount();
    mounted.queryClient?.clear();
    mounted.queryClient = undefined;
  });

  afterAll(() => {
    onlineManager.setEventListener(() => () => undefined);
    focusManager.setEventListener(() => () => undefined);
  });

  it("remplace un cache Préparé par received sans nouvelle déclaration", async () => {
    const queryClient = client();
    queryClient.setQueryData(["driver-cash-collection", 3], prepared);
    let resolveGet: ((value: { data: typeof received }) => void) | undefined;
    get.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveGet = resolve;
        })
    );
    const tree = await mount(queryClient);
    expect(textOf(tree)).not.toContain("À encaisser");
    expect(textOf(tree)).toContain("Réconciliation en cours.");
    await act(async () => {
      resolveGet?.({ data: received });
    });
    await until(tree, "Réglé");
    expect(textOf(tree)).not.toContain("À encaisser");
    expect(post).not.toHaveBeenCalled();
    expect(get).toHaveBeenCalled();
  });

  it("ne renvoie pas la déclaration et retrouve received à la relecture", async () => {
    const queryClient = client();
    const bodies: Record<string, unknown>[] = [];
    get.mockResolvedValueOnce({ data: prepared }).mockResolvedValue({ data: received });
    post.mockImplementation((_url: string, body: Record<string, unknown>) => {
      bodies.push(body);
      return Promise.reject(new Error("réseau"));
    });
    const tree = await mount(queryClient);
    await until(tree, "À encaisser");
    await typeAmount(tree, "45.00");
    await act(async () => {
      const button = tree.root.findByProps({ accessibilityLabel: "Confirmer la réception" });
      button.props.onPress();
      button.props.onPress();
    });
    await until(tree, "Aucune seconde réception automatique");
    expect(textOf(tree)).toContain("Réponse serveur incertaine.");
    expect(textOf(tree)).not.toContain("À encaisser");
    expect(tree.root.findAll((node) => node.props?.accessibilityLabel === "Confirmer la réception")).toHaveLength(0);
    expect(bodies).toHaveLength(1);

    await press(tree, "Relire le dossier");
    await until(tree, "Réglé");
    expect(bodies).toHaveLength(1);
    expect(textOf(tree)).toContain("Réglé");
  });

  it("réutilise la même clé si le serveur est encore préparé", async () => {
    const queryClient = client();
    const bodies: Record<string, unknown>[] = [];
    get.mockResolvedValue({ data: prepared });
    post.mockImplementation((_url: string, body: Record<string, unknown>) => {
      bodies.push(body);
      if (bodies.length === 1) return Promise.reject(new Error("réseau"));
      return Promise.resolve({ data: received });
    });
    const tree = await mount(queryClient);
    await until(tree, "À encaisser");
    await typeAmount(tree, "45.00");
    await press(tree, "Confirmer la réception");
    await until(tree, "Aucune seconde réception automatique");
    await press(tree, "Relire le dossier");
    await until(tree, "À encaisser");
    await typeAmount(tree, "45.00");
    await press(tree, "Confirmer la réception");
    await until(tree, "Réglé");
    expect(bodies).toHaveLength(2);
    expect(bodies[0].idempotency_key).toBe(bodies[1].idempotency_key);
  });

  it("reste incertain si la relecture échoue, sans second envoi", async () => {
    const queryClient = client();
    get.mockResolvedValueOnce({ data: prepared }).mockRejectedValue(new Error("réseau"));
    post.mockRejectedValue(new Error("réseau"));
    const tree = await mount(queryClient);
    await until(tree, "À encaisser");
    await typeAmount(tree, "45.00");
    await press(tree, "Confirmer la réception");
    await until(tree, "Aucune seconde réception automatique");
    await press(tree, "Relire le dossier");
    await until(tree, "Erreur de consultation");
    expect(textOf(tree)).not.toContain("À encaisser");
    expect(tree.root.findAll((node) => node.props?.accessibilityLabel === "Confirmer la réception")).toHaveLength(0);
    expect(post).toHaveBeenCalledTimes(1);
    expect(tree.root.findAll((node) => node.props?.accessibilityLabel === "Relire le dossier").length).toBeGreaterThan(0);
  });

  it("retrouve received après fermeture de l'écran", async () => {
    const queryClient = client();
    get.mockResolvedValueOnce({ data: prepared }).mockResolvedValue({ data: received });
    post.mockRejectedValue(new Error("réseau"));
    const tree = await mount(queryClient);
    await until(tree, "À encaisser");
    await typeAmount(tree, "45.00");
    await press(tree, "Confirmer la réception");
    await until(tree, "Aucune seconde réception automatique");
    await act(async () => {
      closeScreen();
    });
    const again = await mount(queryClient);
    expect(textOf(again)).not.toContain("À encaisser");
    await until(again, "Réglé");
    expect(post).toHaveBeenCalledTimes(1);
  });

  it("affiche le refus serveur sans le confondre avec une réponse perdue", async () => {
    const queryClient = client();
    get.mockRejectedValue({
      response: { status: 409, data: { error_code: "institution_facturee" } },
    });
    const tree = await mount(queryClient);
    await until(tree, "le débiteur est une institution");
    expect(textOf(tree)).not.toContain("Aucune seconde réception automatique");
    expect(textOf(tree)).not.toContain("À encaisser");
    expect(post).not.toHaveBeenCalled();
  });
});
