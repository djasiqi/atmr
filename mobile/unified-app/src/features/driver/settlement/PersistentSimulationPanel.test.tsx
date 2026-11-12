import { createElement } from "react";
import TestRenderer, { act } from "react-test-renderer";
import { Text, TextInput } from "react-native";
import { PersistentSimulationPanel } from "./PersistentSimulationPanel";
import { settlementLedgerAccessAllowed } from "./settlementAccess";

const dossier = {
  banner: "SIMULATION — AUCUN PAIEMENT RÉEL",
  amount_chf: "45.00",
  currency: "CHF",
  state: "preparing",
  mode: null,
  modes: ["tap_to_pay", "cash", "invoice"],
  debtor_label: "Patient fictif",
  company_id: 7,
  contractual_basis: "missing",
  document_version: "2026-10-26.v1-draft",
  signature_recorded: false,
  history: [],
};

function texts(tree: TestRenderer.ReactTestRenderer): string {
  return tree.root
    .findAllByType(Text)
    .map((node) => (typeof node.props.children === "string" ? node.props.children : ""))
    .join(" ");
}

describe("panneau de simulation persistant", () => {
  it("refuse une base réelle même si un drapeau local est vrai", () => {
    expect(
      settlementLedgerAccessAllowed({
        dev: true,
        appVariant: "dev",
        recipeDatabase: "atmr",
        serverLedgerEnabled: true,
      })
    ).toBe(false);
    expect(
      settlementLedgerAccessAllowed({
        dev: true,
        appVariant: "dev",
        recipeDatabase: "atmr_etape24",
        serverLedgerEnabled: false,
      })
    ).toBe(false);
    expect(
      settlementLedgerAccessAllowed({
        dev: true,
        appVariant: "dev",
        recipeDatabase: "atmr_etape24",
        serverLedgerEnabled: true,
      })
    ).toBe(true);
  });

  it("envoie la transition au serveur et affiche la bannière", async () => {
    const actions: string[] = [];
    let tree: TestRenderer.ReactTestRenderer;
    await act(async () => {
      tree = TestRenderer.create(
        createElement(PersistentSimulationPanel, {
          dossier,
          onAction: async (action) => {
            actions.push(action);
          },
        })
      );
    });
    expect(texts(tree!)).toContain("SIMULATION — AUCUN PAIEMENT RÉEL");
    expect(texts(tree!)).toContain("45.00 CHF");
    expect(texts(tree!)).toContain("Entreprise bénéficiaire");
    expect(texts(tree!)).toContain("Paiement par carte simulé");
    expect(texts(tree!)).toContain("Monnaie à rendre : 5.00 CHF");
    const button = tree!.root.findByProps({
      accessibilityLabel: "Lancer Tap to Pay simulé",
    });
    await act(async () => {
      button.props.onPress();
    });
    expect(actions).toEqual(["start_tap"]);
    expect(tree!.root.findAllByType(TextInput)).toHaveLength(0);
  });

  it("envoie les espèces, la facture, le refus et l'incertitude au serveur", async () => {
    const sent: { action: string; body: Record<string, unknown> }[] = [];
    const tree = await renderPanel(dossier, async (action, body) => {
      sent.push({ action, body });
    });
    await press(tree, "Espèces 50 francs monnaie 5 confirmée");
    await press(tree, "Information préalable absente");
    await press(tree, "Acceptation simulée");
    await press(tree, "Refus de signer");
    await press(tree, "Incapacité physique");
    await press(tree, "Difficulté de compréhension");
    await press(tree, "Représentant non habilité");
    await press(tree, "Payeur contesté");
    expect(sent.map((item) => item.action)).toEqual([
      "declare_cash",
      "invoice_event",
      "invoice_event",
      "invoice_event",
      "invoice_event",
      "invoice_event",
      "invoice_event",
      "invoice_event",
    ]);
    expect(sent[0].body).toMatchObject({
      tendered_chf: "50.00",
      change_chf: "5.00",
      change_confirmed: true,
    });
    expect(texts(tree)).toContain("Aucune signature n'est enregistrée.");
    expect(texts(tree)).toContain("aucune valeur juridique");
    expect(texts(tree)).toContain("Ce n'est pas une créance définitive.");
  });

  it("affiche un refus, une incertitude et la reprise sans second débit", async () => {
    const calls: string[] = [];
    const refused = await renderPanel(
      { ...dossier, state: "simulated_refused", modes: ["tap_to_pay", "cash", "invoice"] },
      async (action) => {
        calls.push(action);
      }
    );
    expect(texts(refused)).toContain("Refus simulé");
    const uncertain = await renderPanel(
      {
        ...dossier,
        state: "uncertain",
        mode: "tap_to_pay",
        modes: [],
        history: [{ action: "resolve_tap", to_state: "uncertain" }],
      },
      async (action) => {
        calls.push(action);
        if (action === "recover_tap") throw new Error("réseau");
      }
    );
    expect(texts(uncertain)).toContain("Résultat incertain");
    expect(texts(uncertain)).toContain("Aucun nouveau paiement n'est proposé.");
    expect(uncertain.root.findAllByProps({ accessibilityLabel: "Lancer Tap to Pay simulé" })).toHaveLength(0);
    await press(uncertain, "Relire le résultat serveur");
    expect(texts(uncertain)).toContain("Aucun second débit");
    expect(calls.filter((action) => action === "recover_tap")).toEqual(["recover_tap"]);
  });

  it("ignore un second appui pendant la progression et revient à la mission", async () => {
    let release: (() => void) | undefined;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    const actions: string[] = [];
    let back = 0;
    const tree = await renderPanel(
      dossier,
      () => {
        actions.push("start_tap");
        return gate;
      },
      () => {
        back += 1;
      }
    );
    const button = tree.root.findByProps({
      accessibilityLabel: "Lancer Tap to Pay simulé",
    });
    await act(async () => {
      button.props.onPress();
      button.props.onPress();
    });
    expect(actions).toEqual(["start_tap"]);
    await act(async () => {
      release?.();
    });
    await press(tree, "Retour à la mission");
    expect(back).toBe(1);
    expect(texts(tree)).not.toContain("position GPS");
  });
});

async function renderPanel(
  value: typeof dossier,
  onAction: (action: string, body: Record<string, unknown>) => Promise<void>,
  onBack?: () => void
) {
  let tree!: TestRenderer.ReactTestRenderer;
  await act(async () => {
    tree = TestRenderer.create(
      createElement(PersistentSimulationPanel, { dossier: value, onAction, onBack })
    );
  });
  return tree;
}

async function press(tree: TestRenderer.ReactTestRenderer, label: string) {
  const button = tree.root.findByProps({ accessibilityLabel: label });
  await act(async () => {
    button.props.onPress();
  });
}
