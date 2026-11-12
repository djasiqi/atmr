import { createElement, type ReactElement } from "react";
import TestRenderer, { act } from "react-test-renderer";
import { StyleSheet, Text, TextInput } from "react-native";
import { DriverCashOperation, type CashDossier, type CashReceptionDraft } from "./DriverCashOperation";
import {
  cashDeclarationMayRetry,
  settlementCashPrepAllowed,
} from "./settlementAccess";

function dossier(overrides: Partial<CashDossier> = {}): CashDossier {
  return {
    booking_id: 45,
    company_name: "Transport fictif",
    debtor_label: "Patient fictif",
    amount_due_chf: "45.00",
    currency: "CHF",
    state: "prepared",
    banner: "PRÉPARATION — ENCAISSEMENT NON ACTIVÉ",
    production_activation: false,
    ...overrides,
  };
}

function render(element: ReactElement) {
  let tree: TestRenderer.ReactTestRenderer;
  act(() => {
    tree = TestRenderer.create(element);
  });
  return tree!;
}

function textOf(tree: TestRenderer.ReactTestRenderer): string {
  return tree.root
    .findAllByType(Text)
    .map((node) => {
      const children = node.props.children;
      return Array.isArray(children) ? children.join("") : String(children ?? "");
    })
    .join(" ");
}

function actions(overrides: Record<string, unknown> = {}) {
  return {
    onConfirmReception: jest.fn(),
    onRefuse: jest.fn(),
    onReread: jest.fn(),
    onHandover: jest.fn(),
    onBack: jest.fn(),
    ...overrides,
  };
}

describe("parcours espèces chauffeur", () => {
  it("verrouille le dû et calcule la monnaie, le solde et le refus", () => {
    const onConfirmReception = jest.fn();
    const onRefuse = jest.fn();
    const tree = render(
      createElement(DriverCashOperation, {
        dossier: dossier(),
        ...actions({ onConfirmReception, onRefuse }),
      })
    );
    const body = textOf(tree);
    expect(body).toContain("45.00");
    expect(body).toContain("Non modifiable");
    expect(body).toContain("pas un encaissement réel");
    expect(tree.root.findAllByType(TextInput)).toHaveLength(1);
    const input = tree.root.findByProps({ accessibilityLabel: "Montant remis par le patient" });
    act(() => input.props.onChangeText("45.00"));
    expect(textOf(tree)).toContain("Monnaie à rendre 0.00 CHF");
    act(() => input.props.onChangeText("50"));
    expect(textOf(tree)).toContain("Monnaie à rendre 5.00 CHF");
    act(() => input.props.onChangeText("20"));
    expect(textOf(tree)).toContain("Solde restant 25.00 CHF");
    expect(textOf(tree)).toContain("Aucune facture");
    const confirm = tree.root.findByProps({ accessibilityLabel: "Confirmer la réception" });
    const styled = confirm.findAll((node) => typeof node.props?.style === "function");
    const flat = StyleSheet.flatten(styled[0].props.style({ pressed: false }));
    expect(flat.minHeight).toBeGreaterThanOrEqual(48);
    act(() => confirm.props.onPress());
    const draft = onConfirmReception.mock.calls[0][0] as CashReceptionDraft;
    expect(draft.disposition).toBe("partial");
    expect(draft.tendered_chf).toBe("20.00");
    expect(onRefuse).not.toHaveBeenCalled();
    act(() => {
      tree.root.findByProps({ accessibilityLabel: "Signaler un refus" }).props.onPress();
    });
    expect(tree.root.findByProps({ accessibilityLabel: "Enregistrer le refus" }).props.disabled).toBe(true);
    act(() => {
      tree.root.findByProps({ accessibilityLabel: "Motif du refus" }).props.onChangeText("Patient refuse");
    });
    act(() => {
      tree.root.findByProps({ accessibilityLabel: "Enregistrer le refus" }).props.onPress();
    });
    expect(onRefuse).toHaveBeenCalledWith("Patient refuse");
  });

  it("ne relance pas une déclaration après une réponse perdue", () => {
    expect(cashDeclarationMayRetry(true)).toBe(false);
    const onExact = jest.fn();
    const onReread = jest.fn();
    const tree = render(
      createElement(DriverCashOperation, {
        dossier: dossier({ state: "uncertain" }),
        notice: "État incertain. Aucune seconde réception.",
        ...actions({ onConfirmReception: onExact, onReread }),
        onHandover: jest.fn(),
        onBack: jest.fn(),
      })
    );
    const reread = tree.root.findByProps({ accessibilityLabel: "Relire le dossier" });
    act(() => reread.props.onPress());
    expect(onReread).toHaveBeenCalledTimes(1);
    expect(onExact).not.toHaveBeenCalled();
    expect(textOf(tree)).toContain("Aucune seconde réception");
  });

  it("laisse déclarer la remise seulement après réception", () => {
    const onHandover = jest.fn();
    const tree = render(
      createElement(DriverCashOperation, {
        dossier: dossier({ state: "received", net_chf: "45.00", handover_state: null }),
        ...actions({ onHandover }),
        onBack: jest.fn(),
      })
    );
    act(() => {
      tree.root.findByProps({ accessibilityLabel: "Déclarer la remise" }).props.onPress();
    });
    expect(onHandover).toHaveBeenCalledTimes(1);
  });

  it("refuse la préparation hors base jetable", () => {
    expect(
      settlementCashPrepAllowed({
        dev: true,
        appVariant: "development",
        recipeDatabase: "lirie_e2e_billing_etape36",
        serverPrepEnabled: true,
      })
    ).toBe(true);
    expect(
      settlementCashPrepAllowed({
        dev: true,
        appVariant: "production",
        recipeDatabase: "lirie_e2e_billing_etape36",
        serverPrepEnabled: true,
      })
    ).toBe(false);
    expect(
      settlementCashPrepAllowed({
        dev: true,
        appVariant: "development",
        recipeDatabase: "atmr",
        serverPrepEnabled: true,
      })
    ).toBe(false);
  });

  it("n'affiche pas Préparé tant que la réponse reste incertaine", () => {
    const tree = render(
      createElement(DriverCashOperation, {
        dossier: dossier(),
        notice: "État incertain. Aucune seconde réception automatique.",
        phase: "uncertain",
        ...actions(),
      })
    );
    const body = textOf(tree);
    expect(body).toContain("Réponse serveur incertaine.");
    expect(body).toContain("Aucune seconde réception automatique");
    expect(body).not.toContain("À encaisser");
    expect(tree.root.findAll((node) => node.props?.accessibilityLabel === "Confirmer la réception")).toHaveLength(0);
    expect(tree.root.findAll((node) => node.props?.accessibilityLabel === "Relire le dossier").length).toBeGreaterThan(0);
  });

  it("annonce l'envoi et la réconciliation sans action financière", () => {
    for (const phase of ["sending", "reconciling"] as const) {
      const tree = render(
        createElement(DriverCashOperation, {
          dossier: dossier(),
          phase,
          ...actions(),
        })
      );
      const body = textOf(tree);
      expect(body).not.toContain("À encaisser");
      expect(tree.root.findAll((node) => node.props?.accessibilityLabel === "Confirmer la réception")).toHaveLength(0);
      expect(tree.root.findAll((node) => node.props?.accessibilityLabel === "Relire le dossier")).toHaveLength(0);
      if (phase === "sending") expect(body).toContain("Envoi en cours. Aucune seconde déclaration.");
      if (phase === "reconciling") expect(body).toContain("Réconciliation en cours.");
    }
  });

  it("propose l'envoi fictif après une réception partielle confirmée", () => {
    const onDispatchReceipt = jest.fn();
    const tree = render(
      createElement(DriverCashOperation, {
        dossier: dossier({
          state: "administrative",
          disposition: "partial",
          amount_received_chf: "20.00",
          balance_chf: "25.00",
          net_chf: "20.00",
          receipt_number: "SPECIMEN-1-15",
          receipt_partial: true,
          receipt_lines: ["SIMULATION — QUITTANCE FICTIVE — NON ÉMISE"],
        }),
        ...actions({ onDispatchReceipt }),
      })
    );
    const body = textOf(tree);
    expect(body).toContain("Réception partielle");
    expect(body).toContain("25.00");
    expect(body).toContain("pas un paiement intégral");
    expect(body).toContain("SPECIMEN-1-15");
    expect(body).not.toContain("paiement intégral confirmé");
    const email = tree.root.findByProps({ accessibilityLabel: "Envoyer par e-mail" });
    act(() => email.props.onPress());
    expect(onDispatchReceipt).toHaveBeenCalledWith("email");
    expect(body).toContain("Aucun message réel n'est envoyé");
    expect(body).not.toContain("Tap to Pay");
  });

  it("affiche 20 CHF encaissés et 25 CHF restants sans recalcul", () => {
    const tree = render(
      createElement(DriverCashOperation, {
        dossier: dossier({
          state: "administrative",
          disposition: "partial",
          recognized_net_chf: "20.00",
          remaining_due_chf: "25.00",
          collect_allowed: true,
          net_chf: "20.00",
          actions: { declare_cash: true, tap_to_pay: false, real_receipt: false },
        }),
        ...actions(),
      })
    );
    const body = textOf(tree);
    expect(body).toContain("20.00 CHF encaissés");
    expect(body).toContain("25.00 CHF restants");
    expect(body).not.toContain("Tap to Pay");
    expect(body).not.toContain("Remboursement");
    expect(tree.root.findAllByType(TextInput).length).toBeGreaterThan(0);
  });

  it("affiche la correction et un solde nul sans nouvel encaissement", () => {
    const corrected = render(
      createElement(DriverCashOperation, {
        dossier: dossier({
          state: "administrative",
          recognized_net_chf: "15.00",
          remaining_due_chf: "30.00",
          collect_allowed: true,
          operations: [{ id: 2, kind: "compensation", display: "Correction -5.00 CHF" }],
          history: [{ action: "declare_handover", to_state: "declared" }],
        }),
        ...actions(),
      })
    );
    expect(textOf(corrected)).toContain("15.00 CHF encaissés");
    expect(textOf(corrected)).toContain("30.00 CHF restants");
    expect(textOf(corrected)).toContain("Correction -5.00 CHF");
    expect(textOf(corrected)).toContain("Remise de caisse");
    expect(textOf(corrected)).not.toContain("Remboursement");

    const done = render(
      createElement(DriverCashOperation, {
        dossier: dossier({
          state: "received",
          recognized_net_chf: "45.00",
          remaining_due_chf: "0.00",
          collect_allowed: false,
          replay: true,
          actions: { declare_cash: false, tap_to_pay: false, real_receipt: false },
        }),
        ...actions(),
      })
    );
    expect(textOf(done)).toContain("45.00 CHF encaissés");
    expect(textOf(done)).toContain("0.00 CHF restants");
    expect(textOf(done)).toContain("Opération déjà enregistrée. Aucune nouvelle écriture.");
    expect(done.root.findAll((node) => node.props?.accessibilityLabel === "Confirmer la réception")).toHaveLength(0);
    expect(textOf(done)).not.toContain("Tap to Pay");
  });
});
