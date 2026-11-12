import { useEffect, useRef, useState } from "react";
import { StyleSheet, View } from "react-native";
import { LirieButton } from "../../../design/foundations/LirieButton";
import { LirieText } from "../../../design/foundations/LirieText";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "../../../core/api/client";
import { DriverPatientOperation, type PatientDossier } from "./DriverPatientOperation";
import {
  consultationFailure,
  financialSendAllowed,
  httpRefusal,
  settlementAttempt,
  settlementPanelPhase,
  useSettlementAttempt,
} from "./settlementReconciliation";

const PRESENTED =
  "Brouillon non approuvé. Cette confirmation n'est pas un paiement reçu et n'est pas une reconnaissance de dette.";
const LOST = "État incertain. Aucun second envoi automatique.";

/**
 * Relie l'écran patient aux routes de préparation.
 * Une réponse perdue ne renvoie pas l'action et n'efface pas sa clé.
 */
export function DriverPatientPrepHost({
  missionId,
  onBack,
  allowOpen = false,
}: {
  missionId: number;
  onBack: () => void;
  allowOpen?: boolean;
}) {
  const client = useQueryClient();
  const scope = `patient:${missionId}`;
  const [notice, setNotice] = useState<string | null>(null);
  const [openLocked, setOpenLocked] = useState(false);
  const [openUncertain, setOpenUncertain] = useState(false);
  const opening = useRef(false);
  const path = `/driver/me/bookings/${missionId}/patient-billing-dossier`;
  const reconcile = useSettlementAttempt(scope);
  const reconcileRef = useRef(reconcile);
  reconcileRef.current = reconcile;

  const dossier = useQuery({
    queryKey: ["driver-patient-billing", missionId],
    queryFn: async () => {
      try {
        const existing = await apiClient.get<PatientDossier>(path);
        return existing.data;
      } catch (error: unknown) {
        const status = (error as { response?: { status?: number } })?.response?.status;
        if (status === 404) return null;
        throw error;
      }
    },
    retry: false,
    staleTime: 0,
    refetchOnMount: "always",
    refetchOnReconnect: "always",
    refetchOnWindowFocus: "always",
  });

  const openDossier = useMutation({
    mutationFn: async () => {
      const { data } = await apiClient.post<PatientDossier>(path, {});
      return data;
    },
    retry: false,
    onSuccess: (data) => {
      client.setQueryData(["driver-patient-billing", missionId], data);
      setNotice(null);
      setOpenUncertain(false);
    },
    onError: (error) => {
      const refusal = httpRefusal(error, LOST);
      if (refusal) {
        setNotice(refusal);
        setOpenLocked(true);
        setOpenUncertain(false);
        return;
      }
      setNotice(LOST);
      setOpenUncertain(true);
    },
  });

  const send = useMutation({
    mutationFn: async (body: Record<string, unknown>) => {
      const { data } = await apiClient.post<PatientDossier>(`${path}/transitions`, body);
      return data;
    },
    retry: false,
    onSuccess: (data) => {
      client.setQueryData(["driver-patient-billing", missionId], data);
      setNotice(null);
      reconcileRef.current.finishConfirmed(data.state === "preparing", false);
    },
    onError: (error) => {
      const refusal = httpRefusal(error, LOST);
      if (refusal) {
        setNotice(refusal);
        reconcileRef.current.finishRefused();
        return;
      }
      setNotice(LOST);
      reconcileRef.current.finishLost();
    },
  });

  useEffect(() => {
    if (!dossier.isFetchedAfterMount || dossier.isFetching) return;
    if (!reconcileRef.current.consumeMountPending()) return;
    if (dossier.isError || dossier.isRefetchError || !dossier.data) {
      const failure = consultationFailure(dossier.error, LOST);
      setNotice(failure.text);
      if (failure.refusal) reconcileRef.current.finishRefused();
      else reconcileRef.current.finishRereadFailed();
      return;
    }
    setNotice(null);
    reconcileRef.current.finishConfirmed(dossier.data.state === "preparing", true);
  }, [
    dossier.data,
    dossier.error,
    dossier.isError,
    dossier.isFetchedAfterMount,
    dossier.isFetching,
    dossier.isRefetchError,
  ]);

  const openOnServer = dossier.data?.state === "preparing";
  const failed = Boolean(dossier.isError || dossier.isRefetchError);
  const failure = failed ? consultationFailure(dossier.error, LOST) : null;
  const phase = settlementPanelPhase({
    hasData: Boolean(dossier.data),
    openOnServer,
    fetching: dossier.isFetching,
    fetchedAfterMount: dossier.isFetchedAfterMount,
    failed,
    hold: reconcile.hold,
    refusal: Boolean(failure?.refusal),
  });

  async function reread() {
    if (!reconcileRef.current.beginReread()) return;
    const result = await dossier.refetch();
    if (result.isError || result.error || !result.data) {
      const refusal = httpRefusal(result.error, LOST);
      if (refusal) {
        setNotice(refusal);
        reconcileRef.current.finishRefused();
        return;
      }
      setNotice("Erreur de consultation. Aucune nouvelle opération.");
      reconcileRef.current.finishRereadFailed();
      return;
    }
    setNotice(null);
    reconcileRef.current.finishConfirmed(result.data.state === "preparing", true);
  }

  const absent =
    dossier.isFetchedAfterMount &&
    !dossier.isFetching &&
    !dossier.data &&
    !failed &&
    !reconcile.hold;

  function confirmOpen() {
    if (!allowOpen || openLocked || openUncertain || opening.current || openDossier.isPending) return;
    opening.current = true;
    openDossier.mutate(undefined, {
      onSettled: () => {
        opening.current = false;
      },
    });
  }

  if (absent) {
    return (
      <View accessibilityLabel="Dossier patient absent" style={styles.notice}>
        {notice ? (
          <LirieText variant="message" accessibilityLiveRegion="polite">
            {notice}
          </LirieText>
        ) : (
          <LirieText variant="message">Dossier absent. Aucune facture n'est ouverte.</LirieText>
        )}
        {allowOpen && !openLocked && !openUncertain ? (
          <LirieButton
            variant="secondary"
            busy={openDossier.isPending}
            accessibilityLabel="Préparer le dossier de facturation"
            label="Préparer le dossier de facturation"
            onPress={confirmOpen}
          />
        ) : null}
        {openUncertain ? (
          <LirieButton
            variant="secondary"
            accessibilityLabel="Relire le dossier patient"
            label="Relire le dossier sans renvoyer l'action"
            onPress={() => {
              void dossier.refetch().then((result) => {
                if (result.data) {
                  setNotice(null);
                  setOpenUncertain(false);
                }
              });
            }}
          />
        ) : null}
      </View>
    );
  }

  if (phase === "loading" || phase === "reconciling") {
    const reconciling = phase === "reconciling";
    return (
      <View accessibilityLabel={reconciling ? "Réconciliation du dossier patient" : "Chargement du dossier patient"}>
        <LirieText variant="message" accessibilityLiveRegion="polite">
          {reconciling ? "Réconciliation en cours." : "Chargement du dossier"}
        </LirieText>
      </View>
    );
  }
  if ((phase === "consultation-error" || phase === "refused") && !dossier.data) {
    return (
      <View
        accessibilityLabel={phase === "refused" ? "Refus du serveur" : "Erreur de consultation du dossier patient"}
        style={styles.notice}
      >
        <LirieText variant="message" accessibilityLiveRegion="polite">{failure?.text || notice}</LirieText>
        <LirieButton
          variant="secondary"
          accessibilityLabel="Relire le dossier patient"
          label="Relire le dossier sans renvoyer l'action"
          onPress={() => void reread()}
        />
      </View>
    );
  }
  if (!dossier.data) return null;
  const current = dossier.data;

  function act(name: string, body: Record<string, unknown>) {
    if (!financialSendAllowed(phase)) return;
    const key = reconcile.attempt || settlementAttempt(scope) || `${name}-${missionId}-${current.booking_id}`;
    if (!reconcile.beginSend(key)) return;
    send.mutate({
      ...body,
      idempotency_key: key,
      presented_text: PRESENTED,
      signatory_role: body.signatory_role || "patient",
    });
  }

  const shown = phase === "sending" || phase === "uncertain" || phase === "refused" ? phase : "server";

  return (
    <DriverPatientOperation
      dossier={current}
      notice={notice}
      phase={shown}
      onPresent={() => act("prior", { action: "prior_information" })}
      onAccept={() => act("accept", { action: "acceptance" })}
      onRefuse={() => act("refuse", { action: "refusal" })}
      onUnable={() => act("unable", { action: "unable" })}
      onNotUnderstood={() => act("unclear", { action: "not_understood" })}
      onRepresentative={(name) =>
        act("representative", {
          action: "representative",
          signatory_role: "representative",
          representative_name: name,
          representative_kind: "declare",
        })
      }
      onReread={() => void reread()}
      onBack={onBack}
    />
  );
}

const styles = StyleSheet.create({
  notice: { gap: 8 },
});
