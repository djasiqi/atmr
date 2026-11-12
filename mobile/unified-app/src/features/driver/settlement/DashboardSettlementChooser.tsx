import { useState } from "react";
import { StyleSheet, View } from "react-native";
import { LirieButton } from "../../../design/foundations/LirieButton";
import { LirieText } from "../../../design/foundations/LirieText";
import { useReduceMotion } from "../../../design/navigation/useReduceMotion";
import { Modal } from "../../../design/ui/LegacyModal";
import { DriverCashPrepHost } from "./DriverCashPrepHost";
import { DriverInvoiceSimulation } from "./DriverInvoiceSimulation";

type Mode = "menu" | "cash" | "invoice";

/**
 * Choix des modalités autorisées par la lecture simulée.
 * Tap to Pay reste visible et inactif : aucun écran, aucun appel.
 */
export function DashboardSettlementChooser({
  visible,
  missionId,
  amountLabel,
  cash,
  invoice,
  followupLabel = null,
  onClose,
}: {
  visible: boolean;
  missionId: number;
  amountLabel: string;
  cash: boolean;
  invoice: boolean;
  followupLabel?: string | null;
  onClose: () => void;
}) {
  const [mode, setMode] = useState<Mode>("menu");
  const reduceMotion = useReduceMotion();

  const dismiss = () => {
    setMode("menu");
    onClose();
  };

  let body = (
    <View accessibilityLabel="Règlement de la mission" style={styles.stack}>
      <LirieText variant="message">Montant exigible</LirieText>
      <LirieText variant="title">{amountLabel}</LirieText>
      <LirieText variant="description">Issu de la réservation</LirieText>
      {cash ? (
        <LirieButton
          variant="primary"
          accessibilityLabel="Espèces"
          label="Espèces"
          onPress={() => setMode("cash")}
        />
      ) : null}
      {invoice ? (
        <LirieButton
          variant="primary"
          accessibilityLabel="Facture"
          label="Facture"
          onPress={() => setMode("invoice")}
        />
      ) : null}
      {followupLabel ? <LirieText variant="description">{followupLabel}</LirieText> : null}
      <LirieButton
        variant="secondary"
        disabled
        accessibilityLabel="Tap to Pay, bientôt disponible"
        label="Tap to Pay — Bientôt disponible"
      />
      <LirieText variant="description">Aucune transaction réelle</LirieText>
      <LirieButton
        variant="quiet"
        accessibilityLabel="Retour à la mission"
        label="Retour à la mission"
        onPress={dismiss}
      />
    </View>
  );
  if (mode === "cash" && cash) {
    body = (
      <View style={styles.stack}>
        <DriverCashPrepHost
          missionId={missionId}
          allowOpen
          onBack={() => setMode("menu")}
          onDashboard={dismiss}
        />
      </View>
    );
  } else if (mode === "invoice" && invoice) {
    body = (
      <View style={styles.stack}>
        <DriverInvoiceSimulation
          missionId={missionId}
          writesEnabled
          onBack={() => setMode("menu")}
        />
      </View>
    );
  }

  return (
    <Modal
      visible={visible}
      title="Règlement"
      onClose={dismiss}
      presentation="bottomSheet"
      reduceMotion={reduceMotion}
      footer={null}
      sheetBodyMaxHeightRatio={0.72}
      sheetCoversNavigation
    >
      {body}
    </Modal>
  );
}

const styles = StyleSheet.create({
  stack: { gap: 8, paddingBottom: 48 },
});
