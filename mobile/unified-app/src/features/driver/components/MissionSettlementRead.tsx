import { useState } from "react";
import { StyleSheet, View } from "react-native";
import { useQuery } from "@tanstack/react-query";
import { AppText } from "../../../design/ui/AppText";
import { LirieButton } from "../../../design/foundations/LirieButton";
import { apiClient } from "../../../core/api/client";
import { DriverCashPrepHost } from "../settlement/DriverCashPrepHost";
import { cashSheetClosedLine, exceptionalCashOffer } from "../settlement/dashboardHandover";
import {
  ABSENT,
  type CashRead,
  type PatientRead,
  isTariffAbsence,
  settlementConsultation,
} from "../domain/missionConsultation";

type Props = {
  missionId: number;
  stacked: boolean;
};

function refusalCode(error: unknown): string | null {
  if (!error || typeof error !== "object" || !("response" in error)) return null;
  const data = (error as { response?: { data?: { error_code?: unknown } } }).response?.data;
  return typeof data?.error_code === "string" ? data.error_code : null;
}

async function readOrNull<T>(path: string): Promise<T | null> {
  try {
    const response = await apiClient.get<T>(path);
    return response.data;
  } catch (error: unknown) {
    const status = (error as { response?: { status?: number } }).response?.status;
    if (status === 404) return null;
    throw error;
  }
}

/**
 * Lecture du dossier financier. La feuille espèces ne s'ouvre que sur l'action autorisée.
 */
export function MissionSettlementRead({ missionId, stacked }: Props) {
  const [cashSheet, setCashSheet] = useState(false);
  const contract = useQuery({
    queryKey: ["driver-settlement-read", missionId],
    queryFn: () => readOrNull<Record<string, unknown>>(`/driver/me/bookings/${missionId}/settlement-read`),
    retry: false,
    staleTime: 0,
    refetchOnMount: "always",
  });
  const offer = exceptionalCashOffer(contract.data);
  const reading = contractRecord(contract.data);
  const followup =
    reading?.billing_followup && typeof reading.billing_followup === "object"
      ? (reading.billing_followup as Record<string, unknown>)
      : null;
  const cash = useQuery({
    queryKey: ["driver-cash-collection-read", missionId],
    queryFn: () => readOrNull<CashRead>(`/driver/me/bookings/${missionId}/cash-collection`),
    retry: false,
    staleTime: 0,
    refetchOnMount: "always",
  });
  const patient = useQuery({
    queryKey: ["driver-patient-billing-read", missionId],
    queryFn: () =>
      readOrNull<PatientRead>(`/driver/me/bookings/${missionId}/patient-billing-dossier`),
    retry: false,
    staleTime: 0,
    refetchOnMount: "always",
  });
  const loading = contract.isLoading || cash.isLoading || patient.isLoading;
  const cashCode = refusalCode(cash.error);
  const patientCode = refusalCode(patient.error);
  const contractualPrice = reading?.amount_chf ?? followup?.due_chf;
  const tariffAbsent =
    contractualPrice == null &&
    isTariffAbsence([
      { failed: cash.isError, code: cashCode },
      { failed: patient.isError, code: patientCode },
    ]);
  const notices = [
    cashSheetClosedLine(reading),
    cash.isError && cashCode && cashCode !== "dossier_absent"
      ? `Consultation espèces refusée (${cashCode}).`
      : null,
    patient.isError && patientCode && patientCode !== "dossier_absent"
      ? `Consultation facture refusée (${patientCode}).`
      : null,
    !cash.isError &&
    cash.data &&
    typeof cash.data === "object" &&
    (cash.data as { refusal_reason?: unknown }).refusal_reason === "facture_emise"
      ? "Historique des encaissements consultable. Nouvelle collecte interdite (facture_emise)."
      : null,
  ].filter((line): line is string => Boolean(line));
  const view = settlementConsultation({
    cash: cash.data ?? null,
    patient: patient.data ?? null,
    reservation: reading
      ? {
          price_chf: contractualPrice,
          due_chf: followup?.due_chf,
          recognized_chf: followup?.recognized_chf,
          balance_chf: followup?.balance_chf,
          debtor_label: reading.debtor_label,
          creditor_label: reading.company_name,
          currency: typeof reading.currency === "string" ? reading.currency : "CHF",
        }
      : null,
    failed: contract.isError,
    tariffAbsent,
    notices,
  });

  if (loading) {
    return <AppText variant="bodyMuted">Lecture du règlement…</AppText>;
  }

  if (cashSheet && offer) {
    return (
      <DriverCashPrepHost
        missionId={missionId}
        allowOpen
        exceptional
        onBack={() => setCashSheet(false)}
      />
    );
  }

  return (
    <View accessibilityLabel="Règlement, consultation">
      <Fact label="Prix de la réservation" value={view.price ?? ABSENT} stacked={stacked} />
      <Fact label="Montant dû" value={view.due ?? ABSENT} stacked={stacked} />
      <Fact label="Solde restant" value={view.balance ?? ABSENT} stacked={stacked} />
      <Fact label="Débiteur" value={view.debtor} stacked={stacked} />
      <Fact label="Entreprise créancière" value={view.creditor} stacked={stacked} />
      {offer ? (
        <LirieButton
          variant="secondary"
          label={offer}
          accessibilityLabel={offer}
          onPress={() => setCashSheet(true)}
        />
      ) : null}
      {view.lines.map((line) => (
        <AppText key={line} variant="body">
          {line}
        </AppText>
      ))}
      {view.history.length > 0 ? (
        <View style={styles.history}>
          <AppText variant="label">Historique financier</AppText>
          {view.history.map((item, index) => (
            <AppText key={`${item.label}-${index}`} variant="bodyMuted">
              {item.at ? `${item.label} · ${item.at}` : item.label}
            </AppText>
          ))}
        </View>
      ) : null}
    </View>
  );
}

function contractRecord(payload: unknown): Record<string, unknown> | null {
  if (!payload || typeof payload !== "object") return null;
  const record = payload as Record<string, unknown>;
  const inner = record.data;
  if (inner && typeof inner === "object") return inner as Record<string, unknown>;
  return record;
}

function Fact({ label, value, stacked }: { label: string; value: string; stacked: boolean }) {
  return (
    <View
      style={stacked ? styles.stacked : styles.row}
      accessibilityLabel={`${label}. ${value}`}
    >
      <AppText variant="bodyMuted">{label}</AppText>
      <AppText variant="body" style={stacked ? undefined : styles.value}>
        {value}
      </AppText>
    </View>
  );
}

const styles = StyleSheet.create({
  row: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "flex-start",
    gap: 12,
  },
  stacked: {
    gap: 2,
  },
  value: {
    flex: 1,
    textAlign: "right",
  },
  history: {
    gap: 4,
    marginTop: 8,
  },
});
