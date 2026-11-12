import { useEffect, useRef, useState } from "react";
import { StyleSheet, View } from "react-native";
import { LirieButton } from "../../../design/foundations/LirieButton";
import { LirieText } from "../../../design/foundations/LirieText";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "../../../core/api/client";
import {
  DriverCashOperation,
  type CashDossier,
  type CashPanelPhase,
  type CashReceptionDraft,
} from "./DriverCashOperation";
import {
  consultationFailure,
  financialSendAllowed,
  httpRefusal,
  settlementAttempt,
  settlementPanelPhase,
  useSettlementAttempt,
} from "./settlementReconciliation";

type DriverCashPrepHostProps = {
  missionId: number;
  onBack: () => void;
  /** Le POST d'ouverture n'existe qu'après un appui explicite. */
  allowOpen?: boolean;
  /** Espèces exceptionnelles. Le serveur refuse si le mandat ne les autorise pas. */
  exceptional?: boolean;
  /** Ferme la feuille et revient au tableau de bord. */
  onDashboard?: () => void;
};

const LOST = "État incertain. Aucune seconde réception automatique.";

/**
 * Relie l'écran espèces aux routes de préparation.
 * Une réponse perdue ne déclenche pas une seconde déclaration.
 * Tant que le serveur n'a pas été relu, l'écran ne dit pas « Préparé ».
 */
export function DriverCashPrepHost({
  missionId,
  onBack,
  onDashboard,
  allowOpen = false,
  exceptional = false,
}: DriverCashPrepHostProps) {
  const client = useQueryClient();
  const scope = `cash:${missionId}`;
  const [notice, setNotice] = useState<string | null>(null);
  const [openLocked, setOpenLocked] = useState(false);
  const [openUncertain, setOpenUncertain] = useState(false);
  const opening = useRef(false);
  const autoOpened = useRef(false);
  const path = `/driver/me/bookings/${missionId}/cash-collection`;
  const reconcile = useSettlementAttempt(scope);
  const reconcileRef = useRef(reconcile);
  reconcileRef.current = reconcile;

  const dossier = useQuery({
    queryKey: ["driver-cash-collection", missionId],
    queryFn: async () => {
      try {
        const existing = await apiClient.get<CashDossier>(path);
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
    mutationFn: async (body: Record<string, unknown>) => {
      const { data } = await apiClient.post<CashDossier>(path, body);
      return data;
    },
    retry: false,
    onSuccess: (data) => {
      client.setQueryData(["driver-cash-collection", missionId], data);
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

  const transmit = useMutation({
    mutationFn: async (body: Record<string, unknown>) => {
      const { data } = await apiClient.post<CashDossier>(`${path}/transitions`, body);
      return data;
    },
    retry: false,
    onSuccess: (data) => {
      client.setQueryData(["driver-cash-collection", missionId], data);
      const status = data.receipt_transmission?.status;
      if (status === "captured") setNotice("Message enregistré. Aucun envoi au patient.");
      else if (status === "skipped") setNotice("Aucun envoi. La quittance reste consultable.");
      else if (status === "absent") setNotice("Coordonnées absentes. Aucun message n'a été créé.");
      else if (status === "blocked") setNotice("Destinataire non autorisé. Aucun envoi.");
      else if (status === "failed") setNotice("Échec de capture. Aucune nouvelle réception.");
      else setNotice("Quittance à jour.");
    },
    onError: () => {
      setNotice("La transmission n'a pas abouti. Aucune nouvelle réception.");
    },
  });

  const send = useMutation({
    mutationFn: async (body: Record<string, unknown>) => {
      const { data } = await apiClient.post<CashDossier>(`${path}/transitions`, body);
      return data;
    },
    retry: false,
    onSuccess: (data) => {
      client.setQueryData(["driver-cash-collection", missionId], data);
      setNotice(null);
      reconcileRef.current.finishConfirmed(data.state === "prepared", false);
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
    reconcileRef.current.finishConfirmed(dossier.data.state === "prepared", true);
  }, [
    dossier.data,
    dossier.error,
    dossier.isError,
    dossier.isFetchedAfterMount,
    dossier.isFetching,
    dossier.isRefetchError,
  ]);

  const openOnServer = dossier.data?.state === "prepared";
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
    setOpenUncertain(false);
    reconcileRef.current.finishConfirmed(result.data.state === "prepared", true);
  }

  async function rereadOpen() {
    const result = await dossier.refetch();
    if (result.data) {
      setNotice(null);
      setOpenUncertain(false);
      return;
    }
    const refusal = httpRefusal(result.error, LOST);
    if (refusal) {
      setNotice(refusal);
      setOpenLocked(true);
      setOpenUncertain(false);
      return;
    }
    setNotice(null);
    setOpenUncertain(false);
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
    openDossier.mutate(exceptional ? { exceptional: true } : {}, {
      onSettled: () => {
        opening.current = false;
      },
    });
  }

  useEffect(() => {
    if (!exceptional || !allowOpen || autoOpened.current || openLocked || openUncertain) return;
    if (!dossier.isFetchedAfterMount || dossier.isFetching || dossier.isError || dossier.data) return;
    if (opening.current || openDossier.isPending) return;
    autoOpened.current = true;
    opening.current = true;
    openDossier.mutate(
      { exceptional: true },
      {
        onSettled: () => {
          opening.current = false;
        },
      }
    );
  }, [
    allowOpen,
    dossier.data,
    dossier.isError,
    dossier.isFetchedAfterMount,
    dossier.isFetching,
    exceptional,
    openDossier,
    openLocked,
    openUncertain,
  ]);

  if (absent && exceptional && allowOpen && !openLocked && !openUncertain) {
    return (
      <View accessibilityLabel="Ouverture de la feuille espèces">
        <LirieText variant="message" accessibilityLiveRegion="polite">
          Ouverture de la feuille espèces.
        </LirieText>
      </View>
    );
  }

  if (absent) {
    return (
      <View accessibilityLabel="Dossier espèces absent" style={styles.notice}>
        {notice ? (
          <LirieText variant="message" accessibilityLiveRegion="polite">
            {notice}
          </LirieText>
        ) : (
          <LirieText variant="message">{"Aucune préparation n'est ouverte."}</LirieText>
        )}
        {allowOpen && !openLocked && !openUncertain ? (
          <LirieButton
            variant="secondary"
            busy={openDossier.isPending}
            accessibilityLabel="Préparer le règlement espèces"
            label="Préparer le règlement espèces"
            onPress={confirmOpen}
          />
        ) : null}
        {openUncertain ? (
          <LirieButton
            variant="secondary"
            accessibilityLabel="Relire le dossier"
            label="Relire le dossier sans nouvelle opération"
            onPress={() => void rereadOpen()}
          />
        ) : null}
      </View>
    );
  }

  if (phase === "loading" || phase === "reconciling") {
    const reconciling = phase === "reconciling";
    return (
      <View accessibilityLabel={reconciling ? "Réconciliation du dossier espèces" : "Chargement du dossier espèces"}>
        <LirieText variant="message" accessibilityLiveRegion="polite">
          {reconciling ? "Réconciliation en cours." : "Chargement du dossier"}
        </LirieText>
      </View>
    );
  }
  if ((phase === "consultation-error" || phase === "refused") && !dossier.data) {
    return (
      <View
        accessibilityLabel={phase === "refused" ? "Refus du serveur" : "Erreur de consultation du dossier espèces"}
        style={styles.notice}
      >
        <LirieText variant="message" accessibilityLiveRegion="polite">{failure?.text || notice}</LirieText>
        <LirieButton
          variant="secondary"
          accessibilityLabel="Relire le dossier"
          label="Relire le dossier sans nouvelle opération"
          onPress={() => void reread()}
        />
      </View>
    );
  }
  if (!dossier.data) return null;
  const current = dossier.data;

  function declare(name: string, body: Record<string, unknown>) {
    if (!financialSendAllowed(phase)) return;
    const key = reconcile.attempt || settlementAttempt(scope) || `${name}-${missionId}-${current.booking_id}`;
    if (!reconcile.beginSend(key)) return;
    send.mutate({
      ...body,
      idempotency_key: key,
      ...(exceptional ? { exceptional: true } : {}),
    });
  }

  const shown = (phase === "sending" || phase === "uncertain" || phase === "refused" ? phase : "server") as CashPanelPhase;

  return (
    <DriverCashOperation
      dossier={current}
      notice={notice}
      phase={shown}
      onConfirmReception={(draft: CashReceptionDraft) => declare("reception", { ...draft })}
      onRefuse={(reason) => declare("refus", { action: "refuse_payment", reason })}
      onReread={() => void reread()}
      onHandover={() => declare("handover", { action: "declare_handover" })}
      onPrepareReceipt={() => {
        if (!financialSendAllowed(phase)) return;
        transmit.mutate({
          action: "prepare_receipt",
          idempotency_key: `quittance-${missionId}`,
        });
      }}
      onDispatchReceipt={(channel) => {
        if (!financialSendAllowed(phase)) return;
        transmit.mutate({
          action: "dispatch_receipt",
          channel,
          idempotency_key: `quittance-${channel}-${missionId}`,
        });
      }}
      onBack={onBack}
      onDashboard={onDashboard}
    />
  );
}

const styles = StyleSheet.create({
  notice: { gap: 8 },
});
