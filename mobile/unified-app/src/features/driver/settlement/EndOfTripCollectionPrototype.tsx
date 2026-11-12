import { useState } from "react";
import { Pressable, StyleSheet, TextInput, View } from "react-native";

import { AppText } from "../../../design/responsive";
import { brandPrimary, brandText, surfaceCard } from "../../../design/responsive/colors";
import {
  END_OF_TRIP_COLLECTION_ENABLED,
  INVOICE_SIGNATURE_DISCLAIMER,
  cashReceiptDecision,
  collectionScreenVisible,
  openTapAttempt,
  resolveTapSimulation,
  type TapState,
} from "./endOfTripCollection";

const MODE_LABEL = {
  tap_to_pay: "Carte ou portefeuille",
  cash: "Espèces",
  invoice: "Facture",
} as const;

type Mode = keyof typeof MODE_LABEL;

type Props = {
  enabled?: boolean;
  collectionRequired: boolean;
  ready: boolean;
  modes: Mode[];
  amountChf: string;
  debtorName: string;
  signatureRequired: boolean;
  signatureText: string | null;
  signatureVersion: string | null;
};

/**
 * Maquette non branchée au parcours mission.
 * Le montant et le débiteur viennent du serveur et ne sont pas éditables.
 */
export function EndOfTripCollectionPrototype({
  enabled = END_OF_TRIP_COLLECTION_ENABLED,
  collectionRequired,
  ready,
  modes,
  amountChf,
  debtorName,
  signatureRequired,
  signatureText,
  signatureVersion,
}: Props) {
  const [mode, setMode] = useState<Mode | null>(null);
  const [tap, setTap] = useState<TapState>("ready");
  const [attemptId, setAttemptId] = useState<string | null>(null);
  const [received, setReceived] = useState("");
  const [cashConfirmed, setCashConfirmed] = useState(false);
  const [exception, setException] = useState<string | null>(null);

  if (!collectionScreenVisible(enabled, collectionRequired, ready)) {
    return null;
  }

  const cash = cashReceiptDecision(amountChf, received, cashConfirmed);

  const startTap = () => {
    const nextId = attemptId ?? `simulation-${Date.now().toString(36)}`;
    const opened = openTapAttempt(tap, attemptId, nextId);
    setTap(opened.state);
    setAttemptId(opened.attemptId);
  };

  return (
    <View style={styles.page}>
      <AppText variant="screenTitle">Règlement</AppText>
      <AppText variant="body">
        {amountChf} CHF · {debtorName}
      </AppText>
      <AppText variant="bodyMuted">
        {"Le tarif et le payeur sont ceux de la réservation. Cette étape est distincte de l'attestation du trajet."}
      </AppText>
      <View style={styles.row}>
        {modes.map((item) => (
          <Pressable
            accessibilityRole="button"
            key={item}
            onPress={() => setMode(item)}
            style={[styles.choice, mode === item && styles.choiceOn]}
          >
            <AppText variant="label">{MODE_LABEL[item]}</AppText>
          </Pressable>
        ))}
      </View>
      {mode === "tap_to_pay" ? (
        <View style={styles.panel}>
          <AppText variant="body">Simulation, sans débit.</AppText>
          <AppText variant="bodyMuted">{labelForTap(tap)}</AppText>
          <Pressable accessibilityRole="button" onPress={startTap} style={styles.primary}>
            <AppText variant="label" style={styles.primaryLabel}>
              Lancer la simulation
            </AppText>
          </Pressable>
          {tap === "in_progress" ? (
            <View style={styles.row}>
              <Pressable
                accessibilityRole="button"
                onPress={() => setTap(resolveTapSimulation(tap, attemptId, "confirmed").state)}
              >
                <AppText variant="label">Simuler un accord</AppText>
              </Pressable>
              <Pressable
                accessibilityRole="button"
                onPress={() => setTap(resolveTapSimulation(tap, attemptId, "refused").state)}
              >
                <AppText variant="label">Simuler un refus</AppText>
              </Pressable>
              <Pressable
                accessibilityRole="button"
                onPress={() => setTap(resolveTapSimulation(tap, attemptId, "uncertain").state)}
              >
                <AppText variant="label">Simuler une réponse perdue</AppText>
              </Pressable>
            </View>
          ) : null}
        </View>
      ) : null}
      {mode === "cash" ? (
        <View style={styles.panel}>
          <AppText variant="body">Montant attendu : {amountChf} CHF</AppText>
          <TextInput
            accessibilityLabel="Montant d'espèces reçu"
            keyboardType="decimal-pad"
            onChangeText={(value) => {
              setReceived(value);
              setCashConfirmed(false);
            }}
            placeholder="Montant reçu"
            style={styles.input}
            value={received}
          />
          <Pressable
            accessibilityRole="button"
            onPress={() => setCashConfirmed(true)}
            style={styles.primary}
          >
            <AppText variant="label" style={styles.primaryLabel}>
              Confirmer les espèces reçues
            </AppText>
          </Pressable>
          <AppText variant="bodyMuted">{labelForCash(cash)}</AppText>
        </View>
      ) : null}
      {mode === "invoice" ? (
        <View style={styles.panel}>
          <AppText variant="body">Destinataire : {debtorName}</AppText>
          <AppText variant="body">Montant : {amountChf} CHF</AppText>
          {signatureRequired && signatureText ? (
            <>
              <AppText variant="body">{INVOICE_SIGNATURE_DISCLAIMER}</AppText>
              <AppText variant="caption">Version {signatureVersion}</AppText>
              <AppText variant="body">{signatureText}</AppText>
            </>
          ) : (
            <AppText variant="bodyMuted">
              {"Aucune signature de facturation n'est demandée."}
            </AppText>
          )}
        </View>
      ) : null}
      <Pressable accessibilityRole="button" onPress={() => setException("cannot_pay")}>
        <AppText variant="label">Le client ne peut pas payer</AppText>
      </Pressable>
      <Pressable accessibilityRole="button" onPress={() => setException("signature_refused")}>
        <AppText variant="label">Signature refusée</AppText>
      </Pressable>
      <Pressable accessibilityRole="button" onPress={() => setException("signature_unable")}>
        <AppText variant="label">Signature impossible</AppText>
      </Pressable>
      {exception ? (
        <AppText variant="body">
          {"Incident enregistré dans cette maquette. Le paiement n'est pas marqué reçu et la course n'est pas annulée."}
        </AppText>
      ) : null}
    </View>
  );
}

function labelForTap(state: TapState): string {
  switch (state) {
    case "ready":
      return "Prêt à simuler.";
    case "in_progress":
      return "Simulation en cours.";
    case "confirmed":
      return "Paiement simulé confirmé.";
    case "refused":
      return "Paiement simulé refusé.";
    case "uncertain":
      return "Résultat incertain. Aucun second débit.";
    case "already_done":
      return "Paiement déjà effectué.";
    default:
      return "";
  }
}

function labelForCash(status: "missing" | "declared" | "difference"): string {
  if (status === "declared") return "Espèces déclarées pour ce montant.";
  if (status === "difference") return "Écart constaté. Ce n'est pas un paiement électronique.";
  return "Le choix « Espèces » n'est pas encore une réception.";
}

const styles = StyleSheet.create({
  page: { padding: 16, gap: 12 },
  row: { flexDirection: "row", flexWrap: "wrap", gap: 8 },
  choice: {
    borderWidth: 1,
    borderColor: "#9CA3AF",
    borderRadius: 12,
    paddingVertical: 12,
    paddingHorizontal: 14,
    backgroundColor: surfaceCard,
  },
  choiceOn: { borderColor: brandPrimary },
  panel: { gap: 8, padding: 12, borderRadius: 12, backgroundColor: surfaceCard },
  primary: {
    backgroundColor: brandPrimary,
    borderRadius: 12,
    padding: 14,
    alignItems: "center",
  },
  primaryLabel: { color: "#FFFFFF" },
  input: {
    borderWidth: 1,
    borderColor: "#9CA3AF",
    borderRadius: 12,
    padding: 12,
    color: brandText,
    fontSize: 18,
  },
});
