import { useState } from "react";
import { StyleSheet, TextInput, View } from "react-native";
import { LirieButton } from "../../../design/foundations/LirieButton";
import { LirieLabelBadge } from "../../../design/foundations/LirieStatusBadge";
import { LirieText } from "../../../design/foundations/LirieText";
import { lirieColor, lirieControl, lirieRadius, lirieSpace } from "../../../design/foundations/tokens";

export type PatientDossier = {
  banner: string;
  booking_id: number;
  company_name?: string | null;
  debtor_label: string;
  amount_chf: string;
  currency: string;
  contractual_basis: string;
  document_version: string;
  document_approved?: boolean;
  legally_binding?: boolean;
  real_signature_collected?: boolean;
  invoice_issued?: boolean;
  representative_name?: string | null;
  representative_review?: string;
  representative_powers_verified?: boolean;
  state: string;
};

const DOCUMENT_TITLE = "Facture préparée";
const CONDITIONS =
  "Cette confirmation n'est pas un paiement reçu et ne remplace pas l'attestation de transport.";

/**
 * Préparation d'acceptation. Aucune image de signature n'est capturée.
 * Le montant affiché n'est pas un champ.
 */
export function DriverPatientOperation({
  dossier,
  notice,
  onPresent,
  onAccept,
  onRefuse,
  onUnable,
  onNotUnderstood,
  onRepresentative,
  onReread,
  onBack,
  phase: propsPhase,
}: {
  dossier: PatientDossier;
  notice: string | null;
  onPresent: () => void;
  onAccept: () => void;
  onRefuse: () => void;
  onUnable: () => void;
  onNotUnderstood: () => void;
  onRepresentative: (name: string) => void;
  onReread: () => void;
  onBack: () => void;
  phase?: "server" | "sending" | "uncertain" | "reconciling" | "refused";
}) {
  const [confirming, setConfirming] = useState(false);
  const [representativeName, setRepresentativeName] = useState("");
  const phase = propsPhase ?? "server";
  const label =
    phase === "sending"
      ? "Envoi en cours. Aucun second envoi."
      : phase === "reconciling"
        ? "Réconciliation en cours."
        : phase === "uncertain"
          ? "Réponse serveur incertaine."
          : phase === "refused"
            ? notice || "Refus du serveur. Aucun second dossier."
            : notice
              ? "Réponse serveur incertaine."
              : dossier.state === "preparing"
                ? "Facture préparée."
                : "Facture préparée.";
  const showActions = phase === "server" && !notice && dossier.state === "preparing";
  const showReread =
    phase === "uncertain" || phase === "refused" || (Boolean(notice) && phase === "server");

  const tone = phase === "refused" ? "danger" : phase === "server" && !notice ? "neutral" : "warning";
  return (
    <View style={styles.panel}>
      <LirieText variant="message">
        {dossier.invoice_issued ? "Facture émise" : "Facture préparée"}
      </LirieText>
      <LirieText variant="title">{DOCUMENT_TITLE}</LirieText>
      <LirieText variant="description">{`Entreprise créancière : ${dossier.company_name || "Entreprise"}`}</LirieText>
      <LirieText variant="description">{`Débiteur : ${dossier.debtor_label}`}</LirieText>
      <LirieText variant="description">{`Transport : ${dossier.booking_id}`}</LirieText>
      <LirieText variant="data" style={styles.amount}>{`Montant annoncé : ${dossier.amount_chf} ${dossier.currency}`}</LirieText>
      <LirieText variant="description">{CONDITIONS}</LirieText>
      <LirieText variant="description">{`Version : ${dossier.document_version}`}</LirieText>
      <LirieText variant="description">Qualité : patient ou représentant déclaré</LirieText>
      <LirieText variant="description">Mandat à signer</LirieText>
      <LirieText variant="description">{`Fondement : ${dossier.contractual_basis}`}</LirieText>
      <LirieText variant="description">{`Revue du représentant : ${dossier.representative_review || "absent"}`}</LirieText>
      {dossier.representative_powers_verified ? null : (
        <LirieText variant="description">Pouvoir de représentation : non vérifié</LirieText>
      )}
      <LirieLabelBadge label={label} tone={tone} live />
      {notice && notice !== label ? <LirieText variant="message">{notice}</LirieText> : null}
      {showReread ? (
        <LirieButton
          variant="secondary"
          accessibilityLabel="Relire le dossier patient"
          label="Relire le dossier sans renvoyer l'action"
          onPress={onReread}
        />
      ) : null}
      {showActions ? (
        <View style={styles.actions}>
          <LirieButton
            variant="secondary"
            accessibilityLabel="Présenter l'information"
            label="Présenter l'information préalable"
            onPress={onPresent}
          />
          {confirming ? (
            <LirieButton
              variant="secondary"
              accessibilityLabel="Confirmer l'acceptation"
              label="Confirmer l'acceptation"
              onPress={() => {
                setConfirming(false);
                onAccept();
              }}
            />
          ) : (
            <LirieButton
              variant="secondary"
              accessibilityLabel="Accepter le document"
              label="Accepter le document présenté"
              onPress={() => setConfirming(true)}
            />
          )}
          <LirieButton variant="secondary" accessibilityLabel="Refuser le document" label="Refuser" onPress={onRefuse} />
          <LirieButton
            variant="secondary"
            accessibilityLabel="Déclarer une impossibilité"
            label="Le patient ne peut pas signer"
            onPress={onUnable}
          />
          <LirieButton
            variant="secondary"
            accessibilityLabel="Document non compris"
            label="Le document n'est pas compris"
            onPress={onNotUnderstood}
          />
          <TextInput
            accessibilityLabel="Nom déclaré du représentant"
            value={representativeName}
            onChangeText={setRepresentativeName}
            placeholder="Nom déclaré du représentant"
            placeholderTextColor={lirieColor.inkMuted}
            style={styles.field}
          />
          <LirieButton
            variant="secondary"
            accessibilityLabel="Déclarer un représentant"
            label="Déclarer un représentant, sans approuver son pouvoir"
            onPress={() => onRepresentative(representativeName)}
          />
        </View>
      ) : null}
      <LirieButton variant="quiet" accessibilityLabel="Revenir à la mission" label="Revenir à la mission" onPress={onBack} />
    </View>
  );
}

const styles = StyleSheet.create({
  panel: {
    gap: lirieSpace.sm,
    padding: lirieSpace.md,
    backgroundColor: lirieColor.white,
    borderColor: lirieColor.border,
    borderWidth: 1,
    borderRadius: lirieRadius.card,
  },
  amount: { fontSize: 22, lineHeight: 28 },
  actions: { gap: lirieSpace.sm },
  field: {
    minHeight: lirieControl.minHeight,
    borderWidth: 1,
    borderColor: lirieColor.border,
    borderRadius: lirieRadius.control,
    paddingHorizontal: lirieSpace.md,
    color: lirieColor.ink,
    backgroundColor: lirieColor.white,
  },
});
