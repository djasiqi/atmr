import { createElement, useState } from "react";
import TestRenderer, { act } from "react-test-renderer";
import { QueryClient, QueryClientProvider, focusManager, onlineManager } from "@tanstack/react-query";
import { timeoutManager } from "@tanstack/query-core";
import { Text, View } from "react-native";
import { SafeAreaProvider } from "react-native-safe-area-context";
import { apiClient } from "../../../core/api/client";
import { useDashboardHandover } from "./DashboardHandoverSettlement";

jest.mock("../../../core/api/client", () => ({
  apiClient: { get: jest.fn(), post: jest.fn() },
}));

jest.mock("../../../design/navigation/useReduceMotion", () => ({
  useReduceMotion: () => false,
}));

const mockNet = { connected: true as boolean | null };
jest.mock("@react-native-community/netinfo", () => ({
  useNetInfo: () => ({ isConnected: mockNet.connected }),
}));

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

const cashRead = {
  kind: "collection_authorized",
  reason: "simulation",
  modes: ["cash", "tap_to_pay", "invoice"],
  simulation: true,
  source: "simulation",
  payment_executable: false,
  persists_payment: false,
  stale: false,
};

const prepared = {
  booking_id: 8,
  company_name: "Recette",
  debtor_label: "Patient",
  amount_due_chf: "45.00",
  currency: "CHF",
  state: "prepared",
  banner: "PRÉPARATION — ENCAISSEMENT NON ACTIVÉ",
  production_activation: false,
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

const safeMetrics = {
  insets: { top: 0, left: 0, right: 0, bottom: 34 },
  frame: { x: 0, y: 0, width: 360, height: 780 },
};

function hosted(client: QueryClient, node: ReturnType<typeof createElement>) {
  return createElement(
    SafeAreaProvider,
    { initialMetrics: safeMetrics },
    createElement(QueryClientProvider, { client }, node)
  );
}

function queryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: { retry: false, gcTime: 0 },
      mutations: { retry: false },
    },
  });
}

function absentError() {
  return Object.assign(new Error("absent"), {
    response: { status: 404, data: { error_code: "dossier_absent" } },
  });
}

function Probe({ status }: { status: string }) {
  const handover = useDashboardHandover(8, status);
  return createElement(
    View,
    null,
    createElement(
      Text,
      null,
      handover.showDestinationStep ? "étape-destination" : "parcours-standard"
    ),
    createElement(
      Text,
      { accessibilityLabel: "Confirmer l'arrêt local", onPress: handover.confirmDestination },
      "confirmer"
    ),
    createElement(
      Text,
      null,
      handover.settlementSettled ? "règlement-traité" : "règlement-ouvert"
    ),
    createElement(
      Text,
      {
        accessibilityLabel: "Signaler un règlement impossible",
        onPress: handover.markSettlementWaived,
      },
      "sortie"
    ),
    handover.panel
  );
}

const mounted: TestRenderer.ReactTestRenderer[] = [];

async function mount(
  status: string,
  client = queryClient()
): Promise<TestRenderer.ReactTestRenderer> {
  let tree!: TestRenderer.ReactTestRenderer;
  await act(async () => {
    tree = TestRenderer.create(hosted(client, createElement(Probe, { status })));
  });
  mounted.push(tree);
  return tree;
}

async function until(tree: TestRenderer.ReactTestRenderer, needle: string) {
  for (let attempt = 0; attempt < 20; attempt += 1) {
    if (textOf(tree).includes(needle)) return;
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 10));
    });
  }
  throw new Error(textOf(tree) || "écran vide");
}

async function settle(tree: TestRenderer.ReactTestRenderer) {
  for (let attempt = 0; attempt < 20; attempt += 1) {
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 10));
    });
    if ((apiClient.get as jest.Mock).mock.calls.length > 0 && !textOf(tree).includes("Lecture")) {
      return;
    }
  }
}

async function press(tree: TestRenderer.ReactTestRenderer, label: string) {
  await act(async () => {
    tree.root.findByProps({ accessibilityLabel: label }).props.onPress();
  });
}

function Harness() {
  const [status, setStatus] = useState("IN_PROGRESS");
  return hosted(
    queryClient(),
    createElement(
      View,
      null,
      createElement(Probe, { status }),
      createElement(
        Text,
        { accessibilityLabel: "Clôturer le harnais", onPress: () => setStatus("COMPLETED") },
        "cloture"
      )
    )
  );
}

describe("règlement à la remise", () => {
  const get = apiClient.get as jest.Mock;
  const post = apiClient.post as jest.Mock;

  beforeEach(() => {
    mockNet.connected = true;
    get.mockReset();
    post.mockReset();
  });

  afterEach(async () => {
    await act(async () => {
      mounted.splice(0).forEach((tree) => tree.unmount());
    });
  });

  afterAll(() => {
    onlineManager.setEventListener(() => () => undefined);
    focusManager.setEventListener(() => () => undefined);
  });

  it("ne lit pas avant le patient à bord", async () => {
    const tree = await mount("EN_ROUTE");
    expect(textOf(tree)).toContain("parcours-standard");
    expect(textOf(tree)).not.toContain("étape-destination");
    expect(textOf(tree)).not.toContain("Je suis à l'arrêt à destination");
    expect(get).not.toHaveBeenCalled();
    expect(post).not.toHaveBeenCalled();
  });

  it("masque l'institution, le déjà payé, la facture, l'aller-retour et la politique absente", async () => {
    const cases = [
      { kind: "collection_forbidden", reason: "institution_payer", modes: [] },
      { kind: "already_paid", reason: "payment_confirmed", modes: [] },
      { kind: "collection_forbidden", reason: "facture_emise", modes: [] },
      { kind: "collection_forbidden", reason: "aller_retour_bloque", modes: [] },
      { kind: "administrative_review", reason: "settlement_policy_missing", modes: [] },
      {
        kind: "collection_authorized",
        reason: "simulation",
        modes: ["tap_to_pay"],
        simulation: true,
        source: "simulation",
      },
    ];
    for (const contract of cases) {
      get.mockReset();
      post.mockReset();
      get.mockResolvedValue({
        data: {
          ...cashRead,
          ...contract,
          simulation: contract.simulation ?? false,
          source: contract.source ?? "backend",
        },
      });
      const tree = await mount("IN_PROGRESS");
      await settle(tree);
      await press(tree, "Confirmer l'arrêt local");
      expect(textOf(tree)).toContain("parcours-standard");
      expect(textOf(tree)).not.toContain("étape-destination");
      expect(textOf(tree)).not.toContain("Préparer le règlement espèces");
      expect(textOf(tree)).not.toContain("Consultation impossible");
      expect(post).not.toHaveBeenCalled();
      await act(async () => {
        tree.unmount();
      });
      mounted.pop();
    }
  });

  it("ouvre les espèces seulement après l'étape, une seule fois", async () => {
    let release: (value: { data: typeof prepared }) => void = () => undefined;
    get.mockImplementation((url: string) => {
      if (String(url).includes("settlement-read")) return Promise.resolve({ data: cashRead });
      return Promise.reject(absentError());
    });
    post.mockImplementation(
      () =>
        new Promise((resolve) => {
          release = resolve;
        })
    );
    const tree = await mount("IN_PROGRESS");
    await until(tree, "étape-destination");
    expect(post).not.toHaveBeenCalled();
    expect(textOf(tree)).not.toContain("Préparer le règlement espèces");
    await press(tree, "Confirmer l'arrêt local");
    await press(tree, "Confirmer l'arrêt local");
    await until(tree, "Espèces");
    expect(textOf(tree)).toContain("Bientôt disponible");
    expect(textOf(tree)).not.toContain("Préparer le règlement espèces");
    const tap = tree.root.findByProps({ accessibilityLabel: "Tap to Pay, bientôt disponible" });
    expect(tap.props.disabled).toBe(true);
    expect(tap.props.onPress).toBeUndefined();
    await press(tree, "Espèces");
    await until(tree, "Préparer le règlement espèces");
    expect(post).not.toHaveBeenCalled();
    await act(async () => {
      const button = tree.root.findByProps({ accessibilityLabel: "Préparer le règlement espèces" });
      button.props.onPress();
      button.props.onPress();
      await Promise.resolve();
      release({ data: prepared });
    });
    expect(post).toHaveBeenCalledTimes(1);
    expect(String(post.mock.calls[0][0])).toContain("/cash-collection");
    await until(tree, "À encaisser");
  });

  it("affiche le refus et ne répète pas l'ouverture", async () => {
    get.mockImplementation((url: string) => {
      if (String(url).includes("settlement-read")) return Promise.resolve({ data: cashRead });
      return Promise.reject(absentError());
    });
    post.mockRejectedValue({
      response: { status: 409, data: { error_code: "decision_obsolete" } },
    });
    const tree = await mount("IN_PROGRESS");
    await until(tree, "étape-destination");
    await press(tree, "Confirmer l'arrêt local");
    await until(tree, "Espèces");
    await press(tree, "Espèces");
    await until(tree, "Préparer le règlement espèces");
    await press(tree, "Préparer le règlement espèces");
    await until(tree, "Refus");
    expect(textOf(tree)).not.toContain("Préparer le règlement espèces");
    expect(post).toHaveBeenCalledTimes(1);
  });

  it("ne renvoie pas une ouverture perdue et retrouve une réception", async () => {
    get.mockImplementation((url: string) => {
      if (String(url).includes("settlement-read")) return Promise.resolve({ data: cashRead });
      return Promise.reject(absentError());
    });
    post.mockRejectedValue(new Error("réseau"));
    const tree = await mount("IN_PROGRESS");
    await until(tree, "étape-destination");
    await press(tree, "Confirmer l'arrêt local");
    await until(tree, "Espèces");
    await press(tree, "Espèces");
    await until(tree, "Préparer le règlement espèces");
    await press(tree, "Préparer le règlement espèces");
    await until(tree, "Aucune seconde réception automatique");
    expect(post).toHaveBeenCalledTimes(1);
    get.mockImplementation((url: string) => {
      if (String(url).includes("settlement-read")) return Promise.resolve({ data: cashRead });
      return Promise.resolve({ data: { ...prepared, state: "received" } });
    });
    await press(tree, "Relire le dossier");
    await until(tree, "Réglé");
    expect(post).toHaveBeenCalledTimes(1);
    expect(textOf(tree)).toContain("Réglé");
  });

  it("retire le règlement quand la course est terminée, sans déclaration", async () => {
    get.mockResolvedValue({ data: cashRead });
    let tree!: TestRenderer.ReactTestRenderer;
    await act(async () => {
      tree = TestRenderer.create(createElement(Harness));
    });
    mounted.push(tree);
    await until(tree, "étape-destination");
    await press(tree, "Confirmer l'arrêt local");
    await act(async () => {
      tree.root.findByProps({ accessibilityLabel: "Clôturer le harnais" }).props.onPress();
    });
    expect(textOf(tree)).toContain("parcours-standard");
    expect(textOf(tree)).not.toContain("étape-destination");
    expect(textOf(tree)).not.toContain("Préparer le règlement espèces");
    expect(post).not.toHaveBeenCalled();
  });

  it("signale une lecture impossible sans inventer l'étape", async () => {
    get.mockRejectedValue(new Error("indisponible"));
    const tree = await mount("IN_PROGRESS");
    await until(tree, "Consultation impossible");
    expect(textOf(tree)).toContain("parcours-standard");
    expect(textOf(tree)).not.toContain("étape-destination");
    expect(post).not.toHaveBeenCalled();
  });

  it("masque la rubrique fermée sans la prendre pour une panne", async () => {
    get.mockRejectedValue({
      response: { status: 404, data: { error: "Rubrique indisponible" } },
    });
    const tree = await mount("IN_PROGRESS");
    await settle(tree);
    expect(textOf(tree)).toContain("parcours-standard");
    expect(textOf(tree)).not.toContain("étape-destination");
    expect(textOf(tree)).not.toContain("Consultation impossible");
    expect(textOf(tree)).not.toContain("Préparer le règlement espèces");
    expect(post).not.toHaveBeenCalled();
    expect(get).toHaveBeenCalled();
  });

  it("retrouve un dossier déjà reçu sans le rouvrir", async () => {
    get.mockImplementation((url: string) => {
      if (String(url).includes("settlement-read")) return Promise.resolve({ data: cashRead });
      return Promise.resolve({ data: { ...prepared, state: "received" } });
    });
    const tree = await mount("IN_PROGRESS");
    await until(tree, "étape-destination");
    await press(tree, "Confirmer l'arrêt local");
    await until(tree, "Espèces");
    await press(tree, "Espèces");
    await until(tree, "Réglé");
    expect(post).not.toHaveBeenCalled();
  });

  it("un mandat qui couvre la facturation laisse le parcours standard", async () => {
    get.mockResolvedValue({
      data: {
        ...cashRead,
        kind: "invoice_authorized",
        modes: ["invoice"],
        amount_chf: "45.00",
      },
    });
    const tree = await mount("IN_PROGRESS");
    await settle(tree);
    expect(textOf(tree)).toContain("parcours-standard");
    expect(textOf(tree)).not.toContain("étape-destination");
    expect(textOf(tree)).not.toContain("Montant exigible");
    expect(post).not.toHaveBeenCalled();
  });

  it("ouvre la feuille sans écriture, puis revient à la mission", async () => {
    get.mockResolvedValue({
      data: { ...cashRead, modes: ["cash", "invoice"], amount_chf: "45.00" },
    });
    const tree = await mount("IN_PROGRESS");
    await until(tree, "étape-destination");
    expect(textOf(tree)).not.toContain("Montant exigible");
    expect(post).not.toHaveBeenCalled();
    await press(tree, "Confirmer l'arrêt local");
    await until(tree, "Montant exigible");
    expect(textOf(tree)).toContain("45.00 CHF");
    expect(textOf(tree)).toContain("Issu de la réservation");
    expect(textOf(tree)).toContain("Espèces");
    expect(textOf(tree)).toContain("Facture");
    expect(textOf(tree)).toContain("Bientôt disponible");
    const tap = tree.root.findByProps({ accessibilityLabel: "Tap to Pay, bientôt disponible" });
    expect(tap.props.disabled).toBe(true);
    expect(tap.props.onPress).toBeUndefined();
    expect(post).not.toHaveBeenCalled();
    await press(tree, "Retour à la mission");
    expect(textOf(tree)).not.toContain("Montant exigible");
    expect(textOf(tree)).toContain("étape-destination");
    expect(post).not.toHaveBeenCalled();
    await press(tree, "Confirmer l'arrêt local");
    await until(tree, "Montant exigible");
    expect(post).not.toHaveBeenCalled();
  });

  it("retrouve un partiel déjà enregistré et n'écrit rien", async () => {
    get.mockImplementation((url: string) => {
      if (String(url).includes("settlement-read")) return Promise.resolve({ data: cashRead });
      if (String(url).includes("cash-collection")) {
        return Promise.resolve({ data: { ...prepared, state: "administrative", disposition: "partial" } });
      }
      return Promise.reject(absentError());
    });
    const tree = await mount("IN_PROGRESS");
    await until(tree, "règlement-traité");
    expect(textOf(tree)).toContain("étape-destination");
    expect(post).not.toHaveBeenCalled();
  });

  it("laisse terminer après un signalement sans déclaration financière", async () => {
    get.mockImplementation((url: string) => {
      if (String(url).includes("settlement-read")) return Promise.resolve({ data: cashRead });
      return Promise.reject(absentError());
    });
    const tree = await mount("IN_PROGRESS");
    await until(tree, "étape-destination");
    expect(textOf(tree)).toContain("règlement-ouvert");
    await press(tree, "Signaler un règlement impossible");
    expect(textOf(tree)).toContain("règlement-traité");
    expect(post).not.toHaveBeenCalled();
  });

  it("masque la facture quand le serveur n'autorise que les espèces", async () => {
    get.mockResolvedValue({
      data: { ...cashRead, modes: ["cash"], amount_chf: "45.00" },
    });
    const tree = await mount("IN_PROGRESS");
    await until(tree, "étape-destination");
    await press(tree, "Confirmer l'arrêt local");
    await until(tree, "Espèces");
    expect(textOf(tree)).not.toContain("Facture");
    expect(textOf(tree)).toContain("Bientôt disponible");
    expect(post).not.toHaveBeenCalled();
  });
});
