import { createElement, type ReactElement } from "react";
import TestRenderer, { act } from "react-test-renderer";
import { readFileSync } from "fs";
import { join } from "path";
import { Text, TextInput } from "react-native";
import { DriverSettlementPanel } from "./DriverSettlementPanel";
import type { DriverSettlementContract } from "./driverSettlementContract";
import { settlementSurface } from "./driverSettlementContract";

function contract(
  overrides: Partial<DriverSettlementContract> = {}
): DriverSettlementContract {
  return {
    kind: "no_collection",
    reason: "established_account",
    modes: [],
    amount_chf: null,
    currency: "CHF",
    payment_scope: "booking",
    simulation: false,
    payment_executable: false,
    persists_payment: false,
    provider: "none",
    stale: false,
    authorization_durable: false,
    recheck_required: true,
    attestation_separate: true,
    completion_independent: true,
    driver_may_edit_amount: false,
    driver_may_edit_payer: false,
    writes_completed_payment: false,
    writes_cash_receipt: false,
    writes_invoice: false,
    writes_financial_signature: false,
    invoice_means_paid: false,
    source: "backend",
    company_id: 10,
    debtor_label: "Patient fictif",
    document_approved: false,
    document_version: null,
    signature_required: false,
    attestation_recorded: false,
    payment_received: false,
    preview_screen: null,
    ...overrides,
  };
}

function simulation(
  modes: DriverSettlementContract["modes"]
): DriverSettlementContract {
  return contract({
    kind: "collection_authorized",
    reason: "simulation",
    modes,
    amount_chf: "45.00",
    simulation: true,
    provider: "simulation",
    source: "simulation",
  });
}

function texts(tree: TestRenderer.ReactTestRenderer): string[] {
  return tree.root
    .findAllByType(Text)
    .map((node) => (typeof node.props.children === "string" ? node.props.children : ""))
    .filter(Boolean);
}

function render(element: ReactElement) {
  let tree: TestRenderer.ReactTestRenderer;
  act(() => {
    tree = TestRenderer.create(element);
  });
  return tree!;
}

function press(tree: TestRenderer.ReactTestRenderer, label: string) {
  act(() => {
    tree.root.findByProps({ accessibilityLabel: label }).props.onPress();
  });
}

describe("écran chauffeur de règlement", () => {
  it("reste discret quand aucun règlement n'est demandé", () => {
    const tree = render(
      createElement(DriverSettlementPanel, {
        contract: contract({ kind: "collection_forbidden", reason: "institution_payer" }),
      })
    );
    const body = texts(tree).join(" ");
    expect(body).toContain("Aucun règlement sur place");
    expect(body).not.toContain("Tap to Pay");
    expect(tree.root.findAllByType(TextInput)).toHaveLength(0);
  });

  it("signale un client déjà payé sans bouton d'encaissement", () => {
    const tree = render(
      createElement(DriverSettlementPanel, {
        contract: contract({
          kind: "already_paid",
          reason: "payment_confirmed",
          payment_received: true,
        }),
      })
    );
    expect(texts(tree).join(" ")).toContain("déjà payé");
    expect(texts(tree).join(" ")).not.toContain("Espèces");
  });

  it("affiche la vérification administrative sans choix de payeur", () => {
    const tree = render(
      createElement(DriverSettlementPanel, {
        contract: contract({ kind: "administrative_review", reason: "collection_disabled" }),
      })
    );
    const body = texts(tree).join(" ");
    expect(body).toContain("Règlement à vérifier par l'entreprise.");
    expect(body).not.toContain("Tap to Pay");
    expect(body).not.toContain("Espèces");
    expect(body).not.toContain("Facture");
  });

  it("ne montre en simulation que les moyens indiqués", () => {
    const onlyCard = render(
      createElement(DriverSettlementPanel, { contract: simulation(["tap_to_pay"]) })
    );
    expect(texts(onlyCard).join(" ")).toContain("Tap to Pay");
    expect(texts(onlyCard).join(" ")).not.toContain("Espèces");
    expect(texts(onlyCard).join(" ")).toContain("45.00 CHF");
    expect(texts(onlyCard).join(" ")).toContain("Simulation uniquement");
  });

  it("refuse une relance automatique après une réponse incertaine", () => {
    const tree = render(
      createElement(DriverSettlementPanel, { contract: simulation(["tap_to_pay"]) })
    );
    press(tree, "Tap to Pay");
    press(tree, "Confirmer la simulation");
    press(tree, "Lancer la simulation");
    const uncertain = tree.root.findByProps({
      accessibilityLabel: "Résultat simulé : incertain",
    });
    const accepted = tree.root.findByProps({
      accessibilityLabel: "Résultat simulé : accepté",
    });
    act(() => {
      uncertain.props.onPress();
      accepted.props.onPress();
    });
    const body = texts(tree).join(" ");
    expect(body).toContain("Aucune nouvelle tentative automatique");
    expect(body).not.toContain("Paiement simulé accepté");
    expect(body).not.toContain("Réessayer");
  });

  it("distingue des espèces incomplètes sans les enregistrer", () => {
    const tree = render(
      createElement(DriverSettlementPanel, { contract: simulation(["cash"]) })
    );
    press(tree, "Espèces");
    press(tree, "Confirmer la simulation");
    press(tree, "Le montant reçu est différent");
    const body = texts(tree).join(" ");
    expect(body).toContain("Aucune correction arbitraire");
    expect(body).toContain("Aucune réception d'espèces n'est enregistrée.");
    expect(body).not.toContain("COMPLETED");
  });

  it("présente une facture fictive distincte de l'attestation", () => {
    const tree = render(
      createElement(DriverSettlementPanel, {
        contract: contract({
          kind: "invoice_authorized",
          reason: "simulation",
          modes: ["invoice"],
          amount_chf: "45.00",
          simulation: true,
          provider: "simulation",
          source: "simulation",
          signature_required: true,
          attestation_recorded: true,
          document_approved: false,
        }),
      })
    );
    press(tree, "Facture");
    press(tree, "Confirmer la simulation");
    press(tree, "Patient refusant de signer");
    const body = texts(tree).join(" ");
    expect(body).toContain("n'est pas un paiement");
    expect(body).toContain("distincte de l'attestation");
    expect(body).toContain("Le patient refuse de signer.");
    expect(body).toContain("Aucune signature financière n'est enregistrée.");
  });

  it("couvre le chargement, l'erreur, le hors-ligne et le retour", () => {
    const onBack = jest.fn();
    const loading = render(createElement(DriverSettlementPanel, { contract: null, loading: true }));
    expect(texts(loading).join(" ")).toContain("Lecture du règlement");
    const failed = render(
      createElement(DriverSettlementPanel, { contract: null, error: "perdu" })
    );
    expect(texts(failed).join(" ")).toContain("Aucune autorisation n'est conservée");
    const offline = render(
      createElement(DriverSettlementPanel, { contract: simulation(["cash"]), offline: true })
    );
    expect(texts(offline).join(" ")).toContain("Connexion indisponible");
    expect(texts(offline).join(" ")).not.toContain("Espèces");
    const tree = render(
      createElement(DriverSettlementPanel, { contract: simulation(["tap_to_pay"]), onBack })
    );
    press(tree, "Tap to Pay");
    press(tree, "Retour à la mission");
    expect(onBack).not.toHaveBeenCalled();
    expect(texts(tree).join(" ")).toContain("Tap to Pay");
    press(tree, "Retour à la mission");
    expect(onBack).toHaveBeenCalledTimes(1);
  });

  it("cache l'écran quand le backend n'a rien à montrer", () => {
    expect(settlementSurface(null, {})).toBe("hidden");
    const tree = render(createElement(DriverSettlementPanel, { contract: null }));
    expect(tree.toJSON()).toBeNull();
  });

  it("n'est pas branché sur la clôture, l'attestation ou l'incident", () => {
    const root = join(__dirname, "..", "..", "..", "..");
    const files = [
      "app/(app)/(driver)/missions/[missionId].tsx",
      "app/(app)/(driver)/missions/[missionId]/attestation.tsx",
      "src/features/driver/components/ConfirmCompletionModal.tsx",
      "src/features/driver/statusDictionary.ts",
    ];
    for (const file of files) {
      const source = readFileSync(join(root, file), "utf8");
      expect(source).not.toContain("DriverSettlementPanel");
      expect(source).not.toContain("simulated_driver_contract");
    }
  });

  it("garde des boutons lisibles sans champ de prix", () => {
    const tree = render(
      createElement(DriverSettlementPanel, {
        contract: simulation(["tap_to_pay", "cash", "invoice"]),
      })
    );
    const button = tree.root.findByProps({ accessibilityLabel: "Tap to Pay" });
    const style = Array.isArray(button.props.style) ? button.props.style[0] : button.props.style;
    expect(style.minHeight).toBeGreaterThanOrEqual(48);
    expect(tree.root.findAllByType(TextInput)).toHaveLength(0);
    for (const node of tree.root.findAllByType(Text)) {
      expect(node.props.allowFontScaling).not.toBe(false);
    }
  });
});
