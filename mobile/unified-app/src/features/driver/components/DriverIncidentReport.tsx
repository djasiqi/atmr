import { useEffect, useState } from "react";
import { Modal, Pressable, StyleSheet, TextInput, View } from "react-native";
import { AppText } from "../../../design/ui/AppText";
import { reportDriverMissionIncident } from "../api";
import {
  classifyIncidentSendError,
  createIncidentEventId,
  enqueueIncident,
  flushIncidentOutbox,
  incidentRetriesAutomatically,
  listPendingIncidents,
} from "../domain/incidentOutbox";
import { getItemAsync, setItemAsync } from "../../../core/storage/secureStoreCompat";

const OUTBOX_KEY = "driver_incident_outbox_v1";

const secureIncidentStorage = {
  read: () => getItemAsync(OUTBOX_KEY),
  write: (value: string) => setItemAsync(OUTBOX_KEY, value),
};

const CATEGORIES: { code: string; label: string }[] = [
  { code: "panne_vehicule", label: "Panne du véhicule" },
  { code: "accident", label: "Accident ou événement de sécurité" },
  { code: "passager_absent", label: "Passager absent" },
  { code: "refus_prise_en_charge", label: "Refus de prise en charge" },
  { code: "impossibilite", label: "Impossibilité de poursuivre" },
];

type Props = {
  missionId: number;
  driverUserId: string;
  companyPhone?: string | null;
  visible: boolean;
};

const WAITING = "En attente de transmission. Le signalement est conservé sur cet appareil.";
const SENT = "Transmis. La course n'est pas annulée.";
const UNASSIGNED =
  "En attente de transmission. Vous n'êtes plus affecté à cette mission. L'entreprise ne l'a pas reçu.";
const REJECTED =
  "Signalement non enregistré. Le serveur a refusé l'envoi. Il ne sera pas renvoyé automatiquement.";

export function DriverIncidentReport({
  missionId,
  driverUserId,
  companyPhone,
  visible,
}: Props) {
  const [open, setOpen] = useState(false);
  const [pending, setPending] = useState(false);
  const [detail, setDetail] = useState("");
  const [notice, setNotice] = useState<string | null>(null);

  useEffect(() => {
    if (!visible) return undefined;
    let alive = true;
    void (async () => {
      const queued = await listPendingIncidents(secureIncidentStorage, driverUserId, missionId);
      if (!alive || queued.length === 0) return;
      if (!queued.some(incidentRetriesAutomatically)) {
        setNotice(noticeForMission(queued, false));
        return;
      }
      setNotice(WAITING);
      const report = await flushIncidentOutbox(
        secureIncidentStorage,
        driverUserId,
        transmitIncident
      );
      if (!alive) return;
      const stillWaiting = await listPendingIncidents(
        secureIncidentStorage,
        driverUserId,
        missionId
      );
      setNotice(noticeForMission(stillWaiting, report.transmitted.length > 0));
    })();
    return () => {
      alive = false;
    };
  }, [visible, missionId, driverUserId]);

  if (!visible) return null;

  async function send(code: string) {
    setPending(true);
    const precision = detail.trim();
    await enqueueIncident(secureIncidentStorage, {
      clientEventId: createIncidentEventId(),
      bookingId: missionId,
      driverUserId,
      category: code,
      detail: precision || undefined,
      createdAt: new Date().toISOString(),
    });
    setNotice(WAITING);
    setDetail("");
    setOpen(false);
    try {
      const report = await flushIncidentOutbox(
        secureIncidentStorage,
        driverUserId,
        transmitIncident
      );
      const stillWaiting = await listPendingIncidents(
        secureIncidentStorage,
        driverUserId,
        missionId
      );
      setNotice(noticeForMission(stillWaiting, report.transmitted.length > 0));
    } catch {
      setNotice(WAITING);
    } finally {
      setPending(false);
    }
  }

  return (
    <View>
      <Pressable
        onPress={() => setOpen(true)}
        disabled={pending}
        accessibilityRole="button"
        accessibilityLabel="Signaler un incident sans annuler la course"
        style={styles.button}
      >
        <AppText variant="label" style={styles.buttonLabel}>
          Signaler un incident
        </AppText>
      </Pressable>
      {notice ? (
        <AppText variant="caption" style={styles.notice}>
          {notice}
        </AppText>
      ) : null}
      <Modal visible={open} transparent animationType="fade" onRequestClose={() => setOpen(false)}>
        <Pressable style={styles.backdrop} onPress={() => setOpen(false)}>
          <View style={styles.sheet}>
            <AppText variant="sectionTitle">Incident, sans annulation</AppText>
            <AppText variant="caption">
              {"Ce signalement ne remplace pas un appel. S'il reste en attente, l'entreprise ne l'a pas reçu. En cas de panne, appelez le numéro professionnel de votre entreprise."}
            </AppText>
            <AppText variant="caption">
              En cas de danger immédiat pour le passager, contactez les secours au 144. LIRIE ne
              passe pas cet appel.
            </AppText>
            {companyPhone ? (
              <AppText variant="caption">Numéro professionnel : {companyPhone}</AppText>
            ) : null}
            <TextInput
              value={detail}
              onChangeText={setDetail}
              placeholder="Précision éventuelle"
              accessibilityLabel="Précision éventuelle sur l'incident"
              maxLength={180}
              style={styles.detail}
            />
            {CATEGORIES.map((item) => (
              <Pressable
                key={item.code}
                style={styles.option}
                onPress={() => void send(item.code)}
              >
                <AppText variant="body">{item.label}</AppText>
              </Pressable>
            ))}
          </View>
        </Pressable>
      </Modal>
    </View>
  );
}

/** Signalement traçable. N'ouvre aucun dossier d'espèces ni de facture. */
export async function signalSettlementImpossible(
  missionId: number,
  driverUserId: string
): Promise<"sent" | "queued" | "rejected"> {
  const clientEventId = createIncidentEventId();
  await enqueueIncident(secureIncidentStorage, {
    clientEventId,
    bookingId: missionId,
    driverUserId,
    category: "reglement_impossible",
    detail: "Règlement chauffeur impossible. Aucune déclaration financière.",
    createdAt: new Date().toISOString(),
  });
  try {
    const report = await flushIncidentOutbox(
      secureIncidentStorage,
      driverUserId,
      transmitIncident
    );
    if (report.rejected.includes(clientEventId)) return "rejected";
    if (report.transmitted.includes(clientEventId)) return "sent";
    return "queued";
  } catch {
    return "queued";
  }
}

function noticeForMission(
  rows: Awaited<ReturnType<typeof listPendingIncidents>>,
  transmitted: boolean
): string {
  const retryable = rows.filter(incidentRetriesAutomatically);
  if (retryable.some((item) => item.hold === "unassigned")) return UNASSIGNED;
  if (retryable.length > 0) return WAITING;
  if (rows.some((item) => !incidentRetriesAutomatically(item))) return REJECTED;
  return transmitted ? SENT : WAITING;
}

async function transmitIncident(item: {
  bookingId: number;
  category: string;
  clientEventId: string;
  detail?: string;
}): Promise<"ok" | "network" | "unassigned"> {
  try {
    await reportDriverMissionIncident(
      item.bookingId,
      item.category,
      item.clientEventId,
      item.detail
    );
    return "ok";
  } catch (error) {
    const response = (error as { response?: { status?: number; data?: { error?: string } } })
      .response;
    return classifyIncidentSendError(response?.status ?? null, response?.data?.error ?? null);
  }
}

const styles = StyleSheet.create({
  button: {
    minHeight: 44,
    borderRadius: 12,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#7f1d1d",
  },
  buttonLabel: { color: "#fff", fontWeight: "700" },
  notice: { color: "#7f1d1d" },
  backdrop: {
    flex: 1,
    backgroundColor: "rgba(0,0,0,0.4)",
    justifyContent: "flex-end",
  },
  sheet: {
    backgroundColor: "#fff",
    padding: 16,
    gap: 8,
    borderTopLeftRadius: 16,
    borderTopRightRadius: 16,
  },
  option: { minHeight: 44, justifyContent: "center" },
  detail: {
    minHeight: 44,
    borderWidth: 1,
    borderColor: "#cbd5e1",
    borderRadius: 8,
    paddingHorizontal: 12,
  },
});
