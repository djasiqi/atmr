import { createElement } from "react";
import TestRenderer, { act } from "react-test-renderer";
import { Text } from "react-native";
import { ProductionPrepPanel } from "./ProductionPrepPanel";

function texts(tree: TestRenderer.ReactTestRenderer): string {
  return tree.root
    .findAllByType(Text)
    .map((node) => (typeof node.props.children === "string" ? node.props.children : ""))
    .join(" ");
}

const cash = {
  banner: "PRÉPARATION — ENCAISSEMENT NON ACTIVÉ",
  amount_due_chf: "45.00",
  currency: "CHF",
  debtor_label: "Patient fictif",
  company_name: "Entreprise fictive",
  state: "prepared",
  production_activation: false,
};

const patient = {
  banner: "PRÉPARATION — FACTURE NON ÉMISE",
  amount_chf: "45.00",
  debtor_label: "Patient fictif",
  contractual_basis: "missing",
  document_approved: false,
  legally_binding: false,
  real_signature_collected: false,
  invoice_issued: false,
  production_activation: false,
};

describe("préparation espèces et facture", () => {
  it("affiche le montant dû sans le rendre saisissable et refuse l'activation", () => {
    let sent: Record<string, unknown> | null = null;
    let tree: TestRenderer.ReactTestRenderer;
    act(() => {
      tree = TestRenderer.create(
        createElement(ProductionPrepPanel, {
          cash,
          patient,
          onCash: async (body) => {
            sent = body;
          },
          onPatient: async () => undefined,
        })
      );
    });
    const rendered = texts(tree!);
    expect(rendered).toContain("ENCAISSEMENT NON ACTIVÉ");
    expect(rendered).toContain("FACTURE NON ÉMISE");
    expect(rendered).toContain("45.00");
    expect(rendered).toContain("Activation : non");
    expect(rendered).toContain("Mandat à signer");
    expect(tree!.root.findAllByType(Text).some((node) => node.props.editable)).toBe(false);
    const button = tree!.root.findAll(
      (node) => node.props.onPress && String(textsFrom(node)).includes("rendre 5.00")
    )[0];
    act(() => {
      button.props.onPress();
    });
    expect(sent).toMatchObject({
      tendered_chf: "50.00",
      change_chf: "5.00",
      change_confirmed: true,
    });
  });
});

function textsFrom(node: TestRenderer.ReactTestInstance): string {
  return node.findAllByType(Text).map((item) => item.props.children).join(" ");
}
