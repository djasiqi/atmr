import { createElement } from "react";
import { readFileSync } from "fs";
import { join } from "path";
import TestRenderer, { act } from "react-test-renderer";
import { Text } from "react-native";
import { SettlementValidationDemo } from "./SettlementValidationDemo";
import { cashDemoOutcome, reconnectDemo } from "./cashDemo";

function texts(tree: TestRenderer.ReactTestRenderer): string {
  const visible = tree.root
    .findAllByType(Text)
    .map((node) => (typeof node.props.children === "string" ? node.props.children : ""));
  const labels = tree.root
    .findAll(() => true)
    .map((node) =>
      typeof node.props?.accessibilityLabel === "string" ? node.props.accessibilityLabel : ""
    );
  return [...visible, ...labels].join(" ");
}

function press(tree: TestRenderer.ReactTestRenderer, label: string) {
  const node = tree.root.findAll(
    (item) =>
      item.props?.accessibilityLabel === label && typeof item.props.onPress === "function"
  )[0];
  act(() => {
    node.props.onPress();
  });
}

describe("démonstration de validation", () => {
  it("parcourt les espèces sans émettre de reçu", () => {
    let tree: TestRenderer.ReactTestRenderer | undefined;
    act(() => {
      tree = TestRenderer.create(createElement(SettlementValidationDemo));
    });
    expect(texts(tree!)).toContain("Montant dû 45.00 CHF");
    expect(texts(tree!)).toContain("IN_PROGRESS");
    press(tree!, "Patient à bord");
    press(tree!, "Je suis à l'arrêt à destination");
    press(tree!, "Ouvrir le règlement");
    expect(texts(tree!)).toContain("Panneau espèces");
    press(tree!, "Remettre 45 CHF");
    expect(texts(tree!)).toContain("Aucune confirmation");
    press(tree!, "Confirmer 45 CHF");
    expect(texts(tree!)).toContain("Reçu d'espèces — démonstration, non émis");
    expect(texts(tree!)).toContain("DRAFT — PENDING APPROVAL");
    press(tree!, "Pas de monnaie");
    expect(texts(tree!)).toContain("pas un pourboire");
    expect(texts(tree!)).not.toContain("Reçu d'espèces — démonstration, non émis");
    press(tree!, "Rendre les 30 CHF");
    expect(texts(tree!)).toContain("30.00 CHF rendus");
    press(tree!, "Conserver les 30 CHF");
    expect(texts(tree!)).toContain("30.00 CHF détenus");
    expect(texts(tree!)).toContain("Solde dû 45.00 CHF");
    press(tree!, "Annuler les espèces");
    expect(texts(tree!)).not.toContain("30.00 CHF détenus");
    press(tree!, "Fermer le panneau");
    expect(texts(tree!)).not.toContain("Espèces remises");
    press(tree!, "Terminer la mission");
    expect(texts(tree!)).toContain("COMPLETED");
  });

  it("garde la facture distincte du paiement et de la clôture", () => {
    let tree: TestRenderer.ReactTestRenderer | undefined;
    act(() => {
      tree = TestRenderer.create(createElement(SettlementValidationDemo));
    });
    press(tree!, "Je suis à l'arrêt à destination");
    press(tree!, "Ouvrir la facture");
    expect(texts(tree!)).toContain("2026-10-26.draft-2");
    expect(texts(tree!)).toContain("n'est pas un paiement reçu");
    press(tree!, "Refuser de signer");
    expect(texts(tree!)).toContain("Créance conservée");
    press(tree!, "Patient incapable de signer");
    expect(texts(tree!)).toContain("Aucune signature inventée");
    const before = texts(tree!);
    press(tree!, "Retour au dossier");
    expect(texts(tree!)).toContain("IN_PROGRESS");
    expect(before).toContain("IN_PROGRESS");
    press(tree!, "Terminer la mission");
    expect(texts(tree!)).toContain("COMPLETED");
  });

  it("ne transforme pas une saisie hors ligne en paiement", () => {
    let tree: TestRenderer.ReactTestRenderer | undefined;
    act(() => {
      tree = TestRenderer.create(createElement(SettlementValidationDemo));
    });
    press(tree!, "Je suis à l'arrêt à destination");
    press(tree!, "Ouvrir le règlement");
    press(tree!, "Confirmer 45 CHF");
    press(tree!, "Couper le réseau");
    press(tree!, "Confirmer 45 CHF");
    expect(texts(tree!)).toContain("Hors connexion");
    expect(texts(tree!)).not.toContain("Reçu d'espèces — démonstration, non émis");
    press(tree!, "Rétablir le réseau");
    expect(texts(tree!)).not.toContain("paiement confirmé");
    expect(cashDemoOutcome("30", "held").paymentAccepted).toBe(false);
    expect(reconnectDemo(true).paymentAccepted).toBe(false);
  });

  it("n'est pas branchée sur la mission réelle", () => {
    const root = join(__dirname, "..", "..", "..", "..");
    const mission = readFileSync(
      join(root, "app/(app)/(driver)/missions/[missionId].tsx"),
      "utf8"
    );
    const confirm = readFileSync(
      join(root, "src/features/driver/components/ConfirmCompletionModal.tsx"),
      "utf8"
    );
    expect(mission).not.toContain("SettlementValidationDemo");
    expect(confirm).not.toContain("SettlementValidationDemo");
  });
});
