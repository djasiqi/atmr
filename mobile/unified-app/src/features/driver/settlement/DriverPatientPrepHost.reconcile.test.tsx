import { createElement } from "react";
import TestRenderer, { act } from "react-test-renderer";
import { QueryClient, QueryClientProvider, focusManager, notifyManager, onlineManager } from "@tanstack/react-query";
import { timeoutManager } from "@tanstack/query-core";
import { Text } from "react-native";
import { apiClient } from "../../../core/api/client";
import { DriverPatientPrepHost } from "./DriverPatientPrepHost";
import { clearSettlementAttempts } from "./settlementReconciliation";

jest.mock("../../../core/api/client", () => ({
  apiClient: {
    get: jest.fn(),
    post: jest.fn(),
  },
}));

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

const preparing = {
  banner: "PRÉPARATION — FACTURE NON ÉMISE",
  booking_id: 4,
  company_name: "Recette",
  debtor_label: "Patient facture",
  amount_chf: "45.00",
  currency: "CHF",
  contractual_basis: "missing",
  document_version: "2026-10-26.draft-1",
  document_approved: false,
  legally_binding: false,
  real_signature_collected: false,
  invoice_issued: false,
  representative_review: "absent",
  representative_powers_verified: false,
  state: "preparing",
};

const recorded = { ...preparing, state: "administrative", contractual_basis: "acceptance_unapproved" };

function textOf(tree: TestRenderer.ReactTestRenderer): string {
  return tree.root
    .findAllByType(Text)
    .map((node) => {
      const children = node.props.children;
      return Array.isArray(children) ? children.join("") : String(children ?? "");
    })
    .join(" ");
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

describe("réconciliation facture patient", () => {
  const get = apiClient.get as jest.Mock;
  const post = apiClient.post as jest.Mock;

  beforeEach(() => {
    clearSettlementAttempts();
    get.mockReset();
    post.mockReset();
  });

  it("n'accepte pas une seconde fois après une réponse perdue devenue enregistrée", async () => {
    const queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false, staleTime: 30_000, gcTime: Infinity, refetchOnMount: "ifStale" },
        mutations: { retry: 1 },
      },
    });
    const bodies: Record<string, unknown>[] = [];
    get.mockResolvedValueOnce({ data: preparing }).mockResolvedValue({ data: recorded });
    post.mockImplementation((_url: string, body: Record<string, unknown>) => {
      bodies.push(body);
      return Promise.reject(new Error("réseau"));
    });
    let tree!: TestRenderer.ReactTestRenderer;
    await act(async () => {
      tree = TestRenderer.create(
        createElement(
          QueryClientProvider,
          { client: queryClient },
          createElement(DriverPatientPrepHost, { missionId: 4, onBack: jest.fn() })
        )
      );
    });
    await until(tree, "Accepter le document");
    await act(async () => {
      tree.root.findByProps({ accessibilityLabel: "Accepter le document" }).props.onPress();
    });
    await act(async () => {
      tree.root.findByProps({ accessibilityLabel: "Confirmer l'acceptation" }).props.onPress();
    });
    await until(tree, "Aucun second envoi automatique");
    expect(textOf(tree)).not.toContain("Aucune acceptation enregistrée");
    expect(tree.root.findAll((node) => node.props?.accessibilityLabel === "Accepter le document")).toHaveLength(0);
    await act(async () => {
      tree.root.findByProps({ accessibilityLabel: "Relire le dossier patient" }).props.onPress();
    });
    await until(tree, "Facture préparée");
    expect(bodies).toHaveLength(1);
    expect(tree.root.findAll((node) => node.props?.accessibilityLabel === "Accepter le document")).toHaveLength(0);
    expect(textOf(tree)).toContain("Mandat à signer");
    await act(async () => {
      tree.unmount();
    });
    queryClient.unmount();
    queryClient.clear();
  });
});

afterAll(() => {
  onlineManager.setEventListener(() => () => undefined);
  focusManager.setEventListener(() => () => undefined);
});
