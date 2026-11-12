import { useEffect, useState, type ReactNode } from "react";
import { StyleSheet, View } from "react-native";
import { useNetInfo } from "@react-native-community/netinfo";
import { useQuery } from "@tanstack/react-query";
import { LirieButton } from "../../../design/foundations/LirieButton";
import { LirieText } from "../../../design/foundations/LirieText";
import { apiClient } from "../../../core/api/client";
import { DashboardSettlementChooser } from "./DashboardSettlementChooser";
import {
  cashCollectionSettled,
  dashboardAmountLabel,
  billingFollowupLabel,
  dashboardDestinationStepAuthorized,
  dashboardSettlementModes,
  dashboardVerificationRequired,
  invoiceConfirmationRequired,
  handoverReadClosed,
  invoicePreparationStarted,
  settlementRubricClosed,
  verificationNotice,
} from "./dashboardHandover";

type HandoverState = {
  /** Vrai seulement si le serveur autorise une opération affichable. */
  showDestinationStep: boolean;
  verificationRequired: boolean;
  verificationLabel: string | null;
  destinationConfirmed: boolean;
  /** Réception, facture préparée, refus, incertitude, ou sortie explicite. */
  settlementSettled: boolean;
  confirmDestination: () => void;
  /** Rend « Terminer » sans écrire de déclaration financière. */
  markSettlementWaived: () => void;
  followupLabel: string | null;
  panel: ReactNode;
};

/**
 * Décide l'étape « À destination » par la lecture serveur.
 * La lecture ne crée aucun dossier. L'arrêt confirmé reste local.
 */
export function useDashboardHandover(
  missionId: number,
  operationalStatus: string | null
): HandoverState {
  const [haltConfirmed, setHaltConfirmed] = useState(false);
  const [sheetOpen, setSheetOpen] = useState(false);
  const [waived, setWaived] = useState(false);
  const inProgress = (operationalStatus ?? "").trim().toLowerCase() === "in_progress";
  const net = useNetInfo();
  const offline = net.isConnected === false;

  useEffect(() => {
    setHaltConfirmed(false);
    setSheetOpen(false);
    setWaived(false);
  }, [missionId, operationalStatus]);

  const reading = useQuery({
    queryKey: ["driver-settlement-read", missionId, "handover"],
    queryFn: async () => {
      const { data } = await apiClient.get(`/driver/me/bookings/${missionId}/settlement-read`);
      return data;
    },
    enabled: missionId > 0 && !offline,
    retry: false,
    staleTime: 0,
    refetchOnMount: "always",
    refetchOnReconnect: false,
    refetchOnWindowFocus: false,
  });

  const modes = dashboardSettlementModes(reading.data);
  const verificationRequired =
    (reading.isSuccess && dashboardVerificationRequired(reading.data)) ||
    (reading.isError && settlementRubricClosed(reading.error));
  const destinationAuthorized =
    !verificationRequired && reading.isSuccess && dashboardDestinationStepAuthorized(reading.data);
  const authorized = inProgress && destinationAuthorized;
  const cashOutcome = useQuery({
    queryKey: ["driver-cash-collection", missionId, "handover-outcome"],
    queryFn: async () => {
      const { data } = await apiClient.get(`/driver/me/bookings/${missionId}/cash-collection`);
      return data;
    },
    enabled: authorized,
    retry: false,
    staleTime: 0,
    refetchOnMount: "always",
    refetchOnReconnect: false,
    refetchOnWindowFocus: false,
  });
  const invoiceOutcome = useQuery({
    queryKey: ["driver-patient-billing", missionId, "handover-outcome"],
    queryFn: async () => {
      const { data } = await apiClient.get(
        `/driver/me/bookings/${missionId}/patient-billing-dossier`
      );
      return data;
    },
    enabled: authorized,
    retry: false,
    staleTime: 0,
    refetchOnMount: "always",
    refetchOnReconnect: false,
    refetchOnWindowFocus: false,
  });
  const settlementSettled =
    waived ||
    (cashOutcome.isSuccess && cashCollectionSettled(cashOutcome.data)) ||
    (invoiceOutcome.isSuccess && invoicePreparationStarted(invoiceOutcome.data));
  const amount = dashboardAmountLabel(reading.data);
  const closed = reading.isError && handoverReadClosed(reading.error);
  const outage = inProgress && !authorized && !closed && (offline || reading.isError);
  const confirmed = authorized && haltConfirmed;

  let panel: ReactNode = null;
  if (outage) {
    panel = (
      <View accessibilityLabel="Consultation impossible" style={styles.stack}>
        <LirieText variant="message" accessibilityLiveRegion="polite">
          Consultation impossible
        </LirieText>
        <LirieButton
          variant="secondary"
          accessibilityLabel="Relire le règlement"
          label="Relire le règlement"
          onPress={() => {
            if (!offline) void reading.refetch();
          }}
        />
      </View>
    );
  } else if (confirmed && sheetOpen) {
    panel = (
      <DashboardSettlementChooser
        visible={sheetOpen}
        missionId={missionId}
        amountLabel={amount}
        cash={modes.cash}
        invoice={modes.invoice && invoiceConfirmationRequired(reading.data)}
        followupLabel={billingFollowupLabel(reading.data)}
        onClose={() => {
          setSheetOpen(false);
          void cashOutcome.refetch();
          void invoiceOutcome.refetch();
        }}
      />
    );
  }

  return {
    showDestinationStep: destinationAuthorized,
    verificationRequired,
    verificationLabel: verificationRequired
      ? reading.isSuccess
        ? verificationNotice(reading.data)
        : "Vérification du règlement nécessaire"
      : null,
    destinationConfirmed: confirmed,
    settlementSettled,
    confirmDestination: () => {
      if (!authorized) return;
      setHaltConfirmed(true);
      setSheetOpen(true);
    },
    markSettlementWaived: () => {
      if (!authorized || settlementSettled) return;
      setWaived(true);
      setSheetOpen(false);
    },
    followupLabel: billingFollowupLabel(reading.data),
    panel,
  };
}

const styles = StyleSheet.create({
  stack: { gap: 8 },
});
