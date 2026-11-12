import { createElement, type ReactElement } from "react";
import TestRenderer, { act } from "react-test-renderer";
import { Text, TextInput } from "react-native";
import { DriverPatientOperation, type PatientDossier } from "./DriverPatientOperation";

function render(node: ReactElement) {
  let tree: TestRenderer.ReactTestRenderer;
  act(() => {
    tree = TestRenderer.create(node);
  });
  return tree!;
}

function textOf(tree: TestRenderer.ReactTestRenderer): string {
  return tree.root
    .findAllByType(Text)
    .map((node) => {
      const children = node.props.children;
      return typeof children === "string" ? children : "";
    })
    .join("\n");
}

const dossier: PatientDossier = {
  banner: "PRÉPARATION — FACTURE NON ÉMISE",
  booking_id: 21,
  company_name: "Entreprise fictive",
  debtor_label: "Patient fictif",
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

describe("écran facture patient", () => {
  it("affiche le document et demande une confirmation avant l'acceptation fictive", () => {
    const accepted: string[] = [];
    const tree = render(
      createElement(DriverPatientOperation, {
        dossier,
        notice: null,
        onPresent: () => undefined,
        onAccept: () => accepted.push("accept"),
        onRefuse: () => undefined,
        onUnable: () => undefined,
        onNotUnderstood: () => undefined,
        onRepresentative: () => undefined,
        onReread: () => undefined,
        onBack: () => undefined,
      })
    );
    const shown = textOf(tree);
    expect(shown).toContain("Entreprise fictive");
    expect(shown).toContain("Patient fictif");
    expect(shown).toContain("45.00");
    expect(shown).toContain("2026-10-26.draft-1");
    expect(shown).toContain("Mandat à signer");
    expect(tree.root.findAllByType(TextInput)).toHaveLength(1);
    act(() => {
      tree.root.findByProps({ accessibilityLabel: "Accepter le document" }).props.onPress();
    });
    expect(accepted).toHaveLength(0);
    act(() => {
      tree.root
        .findByProps({ accessibilityLabel: "Confirmer l'acceptation" })
        .props.onPress();
    });
    expect(accepted).toEqual(["accept"]);
  });

  it("masque les actions après une réponse perdue", () => {
    const tree = render(
      createElement(DriverPatientOperation, {
        dossier,
        notice: "État incertain. Aucun second envoi automatique.",
        onPresent: () => undefined,
        onAccept: () => undefined,
        onRefuse: () => undefined,
        onUnable: () => undefined,
        onNotUnderstood: () => undefined,
        onRepresentative: () => undefined,
        onReread: () => undefined,
        onBack: () => undefined,
      })
    );
    expect(textOf(tree)).toContain("Aucun second envoi automatique");
    expect(
      tree.root.findAll((node) => node.props?.accessibilityLabel === "Accepter le document")
    ).toHaveLength(0);
    expect(
      tree.root.findAll(
        (node) => node.props?.accessibilityLabel === "Relire le dossier patient"
      ).length
    ).toBeGreaterThan(0);
    expect(textOf(tree)).not.toContain("Aucune acceptation enregistrée");
  });

  it("n'offre plus d'acceptation quand le serveur a enregistré le dossier", () => {
    const tree = render(
      createElement(DriverPatientOperation, {
        dossier: { ...dossier, state: "administrative" },
        notice: null,
        onPresent: () => undefined,
        onAccept: () => undefined,
        onRefuse: () => undefined,
        onUnable: () => undefined,
        onNotUnderstood: () => undefined,
        onRepresentative: () => undefined,
        onReread: () => undefined,
        onBack: () => undefined,
      })
    );
    expect(textOf(tree)).toContain("Facture préparée");
    expect(tree.root.findAll((node) => node.props?.accessibilityLabel === "Accepter le document")).toHaveLength(0);
  });
});
