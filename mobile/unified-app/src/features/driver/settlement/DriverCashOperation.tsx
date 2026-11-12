import { useState } from "react";
import { StyleSheet, TextInput, View } from "react-native";
import { LirieButton } from "../../../design/foundations/LirieButton";
import { LirieLabelBadge } from "../../../design/foundations/LirieStatusBadge";
import { LirieText } from "../../../design/foundations/LirieText";
import { lirieColor, lirieSpace } from "../../../design/foundations/tokens";

export type CashHistoryLine = { action: string; to_state: string };

export type CashOperationLine = {
  id?: number;
  kind?: string;
  display?: string | null;
  net_chf?: string | null;
};

export type CashDossier = {
  booking_id: number;
  id?: number | null;
  company_name?: string | null;
  debtor_label?: string | null;
  amount_due_chf: string;
  amount_received_chf?: string | null;
  balance_chf?: string | null;
  recognized_net_chf?: string | null;
  remaining_due_chf?: string | null;
  collect_allowed?: boolean;
  replay?: boolean;
  allocated_invoice_payment_id?: number | null;
  imputable?: boolean;
  client_operation_key?: string | null;
  operations?: CashOperationLine[];
  actions?: {
    declare_cash?: boolean;
    compensate_cash?: boolean;
    tap_to_pay?: boolean;
    real_receipt?: boolean;
  };
  currency?: string;
  state: string;
  disposition?: string | null;
  tendered_chf?: string | null;
  change_chf?: string | null;
  net_chf?: string | null;
  refusal_reason?: string | null;
  handover_state?: string | null;
  banner?: string;
  production_activation?: boolean;
  invoice_issued?: boolean;
  real_collection?: boolean;
  local_ledger?: boolean;
  receipt_number?: string | null;
  receipt_partial?: boolean;
  receipt_lines?: string[];
  receipt_transmission?: {
    channel?: string | null;
    status?: string | null;
    sent?: boolean;
    destination?: string | null;
  } | null;
  history?: CashHistoryLine[];
};

export type CashReceptionDraft = {
  action: "declare_cash";
  tendered_chf: string;
  change_chf: string;
  change_confirmed: boolean;
  disposition?: "partial";
};

export type CashPanelPhase = "server" | "sending" | "uncertain" | "reconciling" | "refused";

type DriverCashOperationProps = {
  dossier: CashDossier;
  notice?: string | null;
  phase?: CashPanelPhase;
  onConfirmReception: (draft: CashReceptionDraft) => void;
  onRefuse: (reason: string) => void;
  onReread: () => void;
  onHandover: () => void;
  onPrepareReceipt?: () => void;
  onDispatchReceipt?: (channel: "email" | "sms" | "skip") => void;
  onBack: () => void;
  onDashboard?: () => void;
};

const STATE_LABEL: Record<string, string> = {
  prepared: "À encaisser",
  received: "Réglé",
  returned: "Vérification nécessaire",
  administrative: "Vérification nécessaire",
  uncertain: "Vérification nécessaire",
};

function money(value: string): number | null {
  const normalized = value.trim().replace(",", ".");
  if (!/^\d+(\.\d{1,2})?$/.test(normalized)) return null;
  const amount = Number(normalized);
  if (!Number.isFinite(amount) || amount < 0) return null;
  return Math.round(amount * 100) / 100;
}

function cents(value: number): string {
  return value.toFixed(2);
}

function historyLabel(action: string): string {
  if (action === "compensate_cash") return "Correction";
  if (action === "declare_handover") return "Remise de caisse";
  if (action === "declare_cash") return "Encaissement";
  if (action === "refuse_payment") return "Refus";
  return action;
}

function statusLabel(phase: CashPanelPhase, dossier: CashDossier, notice?: string | null): string {
  if (phase === "sending") return "Envoi en cours. Aucune seconde déclaration.";
  if (phase === "reconciling") return "Réconciliation en cours.";
  if (phase === "uncertain") return "Réponse serveur incertaine.";
  if (phase === "refused") return notice || "Refus du serveur. Aucun second dossier.";
  if (dossier.disposition === "partial") {
    return "Partiellement réglé.";
  }
  if (dossier.disposition === "refused") {
    return "Refus enregistré. La mission peut être terminée.";
  }
  return STATE_LABEL[dossier.state] || dossier.state;
}

/**
 * Saisie des espèces. Le montant dû vient de la réservation.
 * Rien ici n'est un encaissement réel.
 */
export function DriverCashOperation({
  dossier,
  notice,
  phase = "server",
  onConfirmReception,
  onRefuse,
  onReread,
  onHandover,
  onPrepareReceipt,
  onDispatchReceipt,
  onBack,
  onDashboard,
}: DriverCashOperationProps) {
  const [tendered, setTendered] = useState("");
  const [refusing, setRefusing] = useState(false);
  const [reason, setReason] = useState("");
  const blocked = phase !== "server";
  const uncertain =
    phase === "uncertain" ||
    phase === "refused" ||
    (!blocked && (dossier.state === "uncertain" || Boolean(notice)));
  const collectAllowed =
    dossier.actions?.tap_to_pay !== true &&
    (dossier.collect_allowed === true ||
      (dossier.collect_allowed == null && dossier.state === "prepared"));
  const open = phase === "server" && collectAllowed && dossier.remaining_due_chf !== "0.00" && !notice;
  const received = phase === "server" && dossier.state === "received";
  const settled = phase === "server" && dossier.state !== "prepared";
  const due = money(dossier.amount_due_chf);
  const given = money(tendered);
  const change = due != null && given != null && given >= due ? given - due : null;
  const balance = due != null && given != null && given > 0 && given < due ? due - given : null;
  const label = statusLabel(phase, dossier, notice);
  const tone = phase === "refused" || dossier.disposition === "refused" ? "danger" : phase === "server" ? "neutral" : "warning";
  const reasonReady = reason.trim().length >= 3;

  return (
    <View accessibilityLabel="Règlement espèces" style={styles.panel}>
      <LirieText variant="title">Espèces</LirieText>
      <LirieText variant="message">{dossier.banner || "PRÉPARATION — ENCAISSEMENT NON ACTIVÉ"}</LirieText>
      <LirieText variant="description">
        {dossier.local_ledger
          ? "Enregistré dans la caisse locale. Aucune transaction bancaire."
          : "Ce n'est pas un encaissement réel."}
      </LirieText>
      <LirieText variant="description">Référence {dossier.booking_id}</LirieText>
      <LirieText variant="message">Montant dû</LirieText>
      <LirieText variant="data" style={styles.amount}>
        {dossier.amount_due_chf} {dossier.currency || "CHF"}
      </LirieText>
      <LirieText variant="description">Issu de la réservation. Non modifiable.</LirieText>
      {dossier.recognized_net_chf ? (
        <View accessibilityLabel="Position espèces">
          <LirieText variant="message">{`${dossier.recognized_net_chf} CHF encaissés`}</LirieText>
          <LirieText variant="message">{`${dossier.remaining_due_chf || "0.00"} CHF restants`}</LirieText>
          {(dossier.operations || []).map((operation) =>
            operation.kind === "compensation" && operation.display ? (
              <LirieText key={String(operation.id)} variant="description">
                {operation.display}
              </LirieText>
            ) : null
          )}
          {dossier.replay ? (
            <LirieText variant="description">Opération déjà enregistrée. Aucune nouvelle écriture.</LirieText>
          ) : null}
        </View>
      ) : null}
      <LirieLabelBadge label={label} tone={tone} live />
      {notice && phase !== "refused" ? <LirieText variant="message">{notice}</LirieText> : null}
      {open ? (
        <View style={styles.actions}>
          <LirieText variant="message">Montant remis par le patient</LirieText>
          <TextInput
            accessibilityLabel="Montant remis par le patient"
            value={tendered}
            onChangeText={setTendered}
            keyboardType="decimal-pad"
            placeholder="0.00"
            style={styles.input}
          />
          {change != null ? (
            <LirieText variant="message">Monnaie à rendre {cents(change)} CHF</LirieText>
          ) : null}
          {balance != null ? (
            <LirieText variant="message">
              {`Solde restant ${cents(balance)} CHF. Aucune facture n'est émise.`}
            </LirieText>
          ) : null}
          <LirieButton
            variant="primary"
            disabled={given == null || given <= 0 || due == null}
            accessibilityLabel="Confirmer la réception"
            label={
              balance != null
                ? `Confirmer la réception partielle de ${cents(given || 0)} CHF`
                : change != null
                  ? `Confirmer la réception, monnaie ${cents(change)}`
                  : "Confirmer la réception"
            }
            onPress={() => {
              if (given == null || given <= 0 || due == null) return;
              if (given < due) {
                onConfirmReception({
                  action: "declare_cash",
                  tendered_chf: cents(given),
                  change_chf: "0.00",
                  change_confirmed: false,
                  disposition: "partial",
                });
                return;
              }
              onConfirmReception({
                action: "declare_cash",
                tendered_chf: cents(given),
                change_chf: cents(given - due),
                change_confirmed: given > due,
              });
            }}
          />
          <LirieButton
            variant="secondary"
            accessibilityLabel="Signaler un refus"
            label="Signaler un refus"
            onPress={() => setRefusing(true)}
          />
          {refusing ? (
            <View style={styles.actions}>
              <TextInput
                accessibilityLabel="Motif du refus"
                value={reason}
                onChangeText={setReason}
                placeholder="Motif obligatoire"
                style={styles.input}
              />
              <LirieButton
                variant="danger"
                disabled={!reasonReady}
                accessibilityLabel="Enregistrer le refus"
                label="Enregistrer le refus"
                onPress={() => {
                  if (!reasonReady) return;
                  onRefuse(reason.trim());
                }}
              />
            </View>
          ) : null}
        </View>
      ) : null}
      {settled && dossier.disposition === "partial" && !dossier.recognized_net_chf ? (
        <View style={styles.actions}>
          <LirieText variant="message">Montant dû {dossier.amount_due_chf} CHF</LirieText>
          <LirieText variant="message">Montant reçu {dossier.amount_received_chf || dossier.net_chf} CHF</LirieText>
          <LirieText variant="message">Solde restant {dossier.balance_chf} CHF</LirieText>
          <LirieText variant="description">{"Dossier conservé pour l'entreprise. Aucune facture."}</LirieText>
        </View>
      ) : null}
      {settled && dossier.disposition === "refused" ? (
        <LirieText variant="message">Motif : {dossier.refusal_reason || "enregistré"}</LirieText>
      ) : null}
      {settled && (received || dossier.disposition === "partial") ? (
        <View accessibilityLabel="Quittance" style={styles.actions}>
          <LirieText variant="message">{"Réception confirmée par le serveur. Quittance enregistrée."}</LirieText>
          {dossier.receipt_partial || (dossier.balance_chf && dossier.balance_chf !== "0.00") ? (
            <LirieText variant="description">
              {`Réception partielle. Solde restant ${dossier.balance_chf || ""} CHF. Ce n'est pas un paiement intégral.`}
            </LirieText>
          ) : null}
          {dossier.receipt_number ? (
            <LirieText variant="description">{`Référence ${dossier.receipt_number}`}</LirieText>
          ) : (
            <LirieButton
              variant="secondary"
              accessibilityLabel="Préparer la quittance"
              label="Préparer la quittance"
              onPress={onPrepareReceipt}
            />
          )}
          {(dossier.receipt_lines || []).map((line) => (
            <LirieText key={line} variant="description">{line}</LirieText>
          ))}
          {dossier.receipt_number && onDispatchReceipt ? (
            <View style={styles.actions}>
              <LirieText variant="description">{"Aucun message réel n'est envoyé. La capture reste dans la recette."}</LirieText>
              <LirieButton
                variant="secondary"
                accessibilityLabel="Envoyer par e-mail"
                label="Envoyer par e-mail"
                onPress={() => onDispatchReceipt("email")}
              />
              <LirieButton
                variant="secondary"
                accessibilityLabel="Envoyer par SMS"
                label="Envoyer par SMS"
                onPress={() => onDispatchReceipt("sms")}
              />
              <LirieButton
                variant="secondary"
                accessibilityLabel="Poursuivre sans envoi"
                label="Poursuivre sans envoi"
                onPress={() => onDispatchReceipt("skip")}
              />
            </View>
          ) : null}
          {dossier.receipt_transmission ? (
            <LirieText variant="description">
              {`Transmission ${dossier.receipt_transmission.status || "enregistrée"}, envoi réel non effectué.`}
            </LirieText>
          ) : null}
        </View>
      ) : null}
      {(dossier.history || []).length > 0 ? (
        <View accessibilityLabel="Historique des espèces">
          {dossier.history?.map((line, index) => (
            <LirieText key={`${line.action}-${index}`} variant="description">
              {historyLabel(line.action)}
            </LirieText>
          ))}
        </View>
      ) : null}
      {uncertain ? (
        <LirieButton
          variant="secondary"
          accessibilityLabel="Relire le dossier"
          label="Relire le dossier sans nouvelle réception"
          onPress={onReread}
        />
      ) : null}
      {received && !dossier.handover_state ? (
        <LirieButton
          variant="secondary"
          accessibilityLabel="Déclarer la remise"
          label="Déclarer la remise au responsable"
          onPress={onHandover}
        />
      ) : null}
      {dossier.handover_state ? <LirieText variant="description">Remise : {dossier.handover_state}</LirieText> : null}
      {settled ? (
        <LirieButton
          variant="primary"
          accessibilityLabel="Retour au tableau de bord"
          label="Retour au tableau de bord"
          onPress={onDashboard || onBack}
        />
      ) : (
        <LirieButton variant="quiet" accessibilityLabel="Retour à la mission" label="Retour à la mission" onPress={onBack} />
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  panel: {
    gap: lirieSpace.sm,
    padding: lirieSpace.md,
    paddingBottom: 48,
    backgroundColor: lirieColor.white,
    borderColor: lirieColor.border,
    borderWidth: 1,
    borderRadius: 12,
  },
  amount: { fontSize: 28, lineHeight: 34 },
  actions: { gap: lirieSpace.sm },
  input: {
    minHeight: 48,
    borderWidth: 1,
    borderColor: lirieColor.border,
    borderRadius: 12,
    paddingHorizontal: 12,
    fontSize: 18,
    color: lirieColor.ink,
    backgroundColor: lirieColor.white,
  },
});
