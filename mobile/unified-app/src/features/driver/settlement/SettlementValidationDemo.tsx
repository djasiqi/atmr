import { useState } from "react";
import { Pressable, StyleSheet, Text, View } from "react-native";
import { cashDemoOutcome, reconnectDemo, type CashDemo } from "./cashDemo";

type Panel = "closed" | "cash" | "invoice";
type Tendered = "0" | "30" | "45" | "50";

/**
 * Démonstration locale. Elle n'est pas montée sur la mission réelle.
 */
export function SettlementValidationDemo() {
  const [status, setStatus] = useState("IN_PROGRESS");
  const [aboard, setAboard] = useState(false);
  const [halt, setHalt] = useState(false);
  const [panel, setPanel] = useState<Panel>("closed");
  const [tendered, setTendered] = useState<Tendered | null>(null);
  const [outcome, setOutcome] = useState<CashDemo | null>(null);
  const [offline, setOffline] = useState(false);
  const [invoiceNote, setInvoiceNote] = useState<string | null>(null);

  const choose = (amount: Tendered, choice: Parameters<typeof cashDemoOutcome>[1]) => {
    if (offline) {
      setOutcome({
        paymentAccepted: false,
        receiptDemo: false,
        heldChf: "0.00",
        balanceDueChf: "45.00",
        administrative: true,
        message: "Hors connexion. Aucun paiement n'est confirmé.",
      });
      return;
    }
    setTendered(amount);
    setOutcome(cashDemoOutcome(amount, choice));
  };

  return (
    <View accessibilityLabel="Démonstration de validation">
      <Text>{status}</Text>
      <Text>{"Montant dû 45.00 CHF"}</Text>
      {aboard ? <Text>{"Patient à bord"}</Text> : null}
      <Pressable
        accessibilityRole="button"
        accessibilityLabel="Patient à bord"
        onPress={() => setAboard(true)}
        style={styles.button}
      >
        <Text>{"Patient à bord"}</Text>
      </Pressable>
      {status === "IN_PROGRESS" && !halt ? (
        <Pressable
          accessibilityRole="button"
          accessibilityLabel="Je suis à l'arrêt à destination"
          onPress={() => setHalt(true)}
          style={styles.button}
        >
          <Text>{"Je suis à l'arrêt à destination"}</Text>
        </Pressable>
      ) : null}
      {halt && panel === "closed" ? (
        <Pressable
          accessibilityRole="button"
          accessibilityLabel="Ouvrir le règlement"
          onPress={() => setPanel("cash")}
          style={styles.button}
        >
          <Text>{"Ouvrir le règlement"}</Text>
        </Pressable>
      ) : null}
      {panel !== "closed" ? (
        <Pressable
          accessibilityRole="button"
          accessibilityLabel="Fermer le panneau"
          onPress={() => setPanel("closed")}
          style={styles.button}
        >
          <Text>{"Fermer le panneau"}</Text>
        </Pressable>
      ) : null}
      <Pressable
        accessibilityRole="button"
        accessibilityLabel="Terminer la mission"
        onPress={() => {
          setStatus("COMPLETED");
          setPanel("closed");
        }}
        style={styles.button}
      >
        <Text>{"Terminer la mission"}</Text>
      </Pressable>
      {panel === "cash" ? (
        <View accessibilityLabel="Panneau espèces">
          <Text>{"DRAFT — PENDING APPROVAL"}</Text>
          <Text>{"Espèces remises"}</Text>
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Remettre 45 CHF"
            onPress={() => choose("45", null)}
            style={styles.button}
          >
            <Text>{"45.00 CHF"}</Text>
          </Pressable>
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Confirmer 45 CHF"
            onPress={() => choose("45", "confirm")}
            style={styles.button}
          >
            <Text>{"Confirmer la réception de 45.00 CHF"}</Text>
          </Pressable>
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Remettre 50 CHF"
            onPress={() => choose("50", null)}
            style={styles.button}
          >
            <Text>{"50.00 CHF, monnaie à confirmer"}</Text>
          </Pressable>
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Confirmer la monnaie rendue"
            onPress={() => choose("50", "change_yes")}
            style={styles.button}
          >
            <Text>{"5.00 CHF rendus"}</Text>
          </Pressable>
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Pas de monnaie"
            onPress={() => choose("50", "change_no")}
            style={styles.button}
          >
            <Text>{"Pas de monnaie"}</Text>
          </Pressable>
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Remettre 30 CHF"
            onPress={() => choose("30", null)}
            style={styles.button}
          >
            <Text>{"30.00 CHF insuffisants"}</Text>
          </Pressable>
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Rendre les 30 CHF"
            onPress={() => choose("30", "returned")}
            style={styles.button}
          >
            <Text>{"Rendre les 30.00 CHF"}</Text>
          </Pressable>
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Conserver les 30 CHF"
            onPress={() => choose("30", "held")}
            style={styles.button}
          >
            <Text>{"Conserver les 30.00 CHF"}</Text>
          </Pressable>
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Ne rien remettre"
            onPress={() => choose("0", null)}
            style={styles.button}
          >
            <Text>{"Ne rien remettre"}</Text>
          </Pressable>
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Annuler les espèces"
            onPress={() => {
              setTendered(null);
              setOutcome(null);
            }}
            style={styles.button}
          >
            <Text>{"Annuler"}</Text>
          </Pressable>
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Couper le réseau"
            onPress={() => setOffline(true)}
            style={styles.button}
          >
            <Text>{"Couper le réseau"}</Text>
          </Pressable>
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Rétablir le réseau"
            onPress={() => {
              const back = reconnectDemo(Boolean(outcome?.receiptDemo));
              setOffline(false);
              setOutcome((current) =>
                current
                  ? {
                      ...current,
                      paymentAccepted: back.paymentAccepted,
                      administrative: current.administrative || back.administrative,
                      receiptDemo: false,
                    }
                  : current
              );
            }}
            style={styles.button}
          >
            <Text>{"Rétablir le réseau"}</Text>
          </Pressable>
          {tendered ? <Text>{`Saisie ${tendered} CHF`}</Text> : null}
          {outcome ? <Text>{outcome.message}</Text> : null}
          {outcome?.receiptDemo ? (
            <Text>{"Reçu d'espèces — démonstration, non émis"}</Text>
          ) : null}
        </View>
      ) : null}
      {halt ? (
        <Pressable
          accessibilityRole="button"
          accessibilityLabel="Ouvrir la facture"
          onPress={() => setPanel("invoice")}
          style={styles.button}
        >
          <Text>{"Ouvrir la facture"}</Text>
        </Pressable>
      ) : null}
      {panel === "invoice" ? (
        <View accessibilityLabel="Document de facturation">
          <Text>{"DRAFT — PENDING APPROVAL"}</Text>
          <Text>{"2026-10-26.draft-2"}</Text>
          <Text>
            {
              "Je confirme avoir pris connaissance des informations de facturation présentées pour le transport indiqué."
            }
          </Text>
          <Text>{"Ce document n'est pas un paiement reçu."}</Text>
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Refuser de signer"
            onPress={() =>
              setInvoiceNote("Signature refusée. Traitement administratif. Créance conservée.")
            }
            style={styles.button}
          >
            <Text>{"Refuser de signer"}</Text>
          </Pressable>
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Patient incapable de signer"
            onPress={() =>
              setInvoiceNote("Signature impossible. Aucune signature inventée.")
            }
            style={styles.button}
          >
            <Text>{"Patient incapable de signer"}</Text>
          </Pressable>
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Retour au dossier"
            onPress={() => setPanel("closed")}
            style={styles.button}
          >
            <Text>{"Retour au dossier"}</Text>
          </Pressable>
          {invoiceNote ? <Text>{invoiceNote}</Text> : null}
        </View>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  button: {
    minHeight: 48,
    justifyContent: "center",
  },
});
