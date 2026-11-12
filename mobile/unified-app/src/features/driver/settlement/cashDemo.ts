/**
 * Règles d'écran pour la démonstration. Elles n'encaissent rien.
 * La décision métier reste dans le domaine Python.
 */

export type CashDemo = {
  paymentAccepted: false;
  receiptDemo: boolean;
  heldChf: string;
  balanceDueChf: string;
  message: string;
  administrative: boolean;
};

export function cashDemoOutcome(
  tendered: "0" | "30" | "45" | "50",
  choice: "returned" | "held" | "change_yes" | "change_no" | "confirm" | null
): CashDemo {
  const unpaid = {
    paymentAccepted: false as const,
    receiptDemo: false,
    heldChf: "0.00",
    balanceDueChf: "45.00",
    administrative: true,
  };
  if (tendered === "0") {
    return { ...unpaid, message: "Aucune espèce remise. Course non payée." };
  }
  if (tendered === "30" && choice === "returned") {
    return {
      ...unpaid,
      message: "30.00 CHF rendus. Aucun encaissement. Solde dû 45.00 CHF.",
    };
  }
  if (tendered === "30" && choice === "held") {
    return {
      ...unpaid,
      heldChf: "30.00",
      message:
        "30.00 CHF détenus, non acceptés comme règlement. Solde dû 45.00 CHF.",
    };
  }
  if (tendered === "30") {
    return {
      ...unpaid,
      message: "Indiquer si les 30.00 CHF sont rendus ou conservés.",
    };
  }
  if (tendered === "45" && choice === "confirm") {
    return {
      paymentAccepted: false,
      receiptDemo: true,
      heldChf: "0.00",
      balanceDueChf: "0.00",
      administrative: false,
      message: "Démonstration de reçu. Document non émis.",
    };
  }
  if (tendered === "50" && choice === "change_yes") {
    return {
      paymentAccepted: false,
      receiptDemo: true,
      heldChf: "0.00",
      balanceDueChf: "0.00",
      administrative: false,
      message: "Monnaie de 5.00 CHF confirmée. Reçu non émis.",
    };
  }
  if (tendered === "50" && choice === "change_no") {
    return {
      ...unpaid,
      message: "Pas de monnaie. Le surplus n'est pas un pourboire.",
    };
  }
  return {
    ...unpaid,
    administrative: false,
    message: "Montant affiché. Aucune confirmation.",
  };
}

export function reconnectDemo(localConfirmed: boolean): {
  paymentAccepted: false;
  administrative: boolean;
} {
  return { paymentAccepted: false, administrative: localConfirmed };
}
