import { StyleSheet, View } from "react-native";
import { LirieButton } from "../../../design/foundations/LirieButton";
import { LirieText } from "../../../design/foundations/LirieText";
import { lirieColor, lirieRadius, lirieSpace } from "../../../design/foundations/tokens";
import { DriverSettlementPanel } from "./DriverSettlementPanel";
import type { MissionPanelState } from "./missionPanelState";

/**
 * Action facultative sur le détail de mission.
 * L'arrêt est confirmé par le chauffeur. Aucun moyen n'est affiché avant
 * l'ouverture, et une décision réelle n'affiche jamais de bouton de paiement.
 */
export function MissionSettlementEntry({
  state,
  onConfirmHalt,
  onOpen,
  onClose,
}: {
  state: MissionPanelState;
  onConfirmHalt: () => void;
  onOpen: () => void;
  onClose: () => void;
}) {
  const active =
    state.showHaltPrompt ||
    state.showSettlementAction ||
    state.visible ||
    state.notice !== null ||
    state.loading ||
    state.offline ||
    state.error !== null;
  if (!active) return null;
  return (
    <View accessibilityLabel="Règlement du transport" style={styles.wrap}>
      <LirieText variant="title">Règlement</LirieText>
      {state.consultation && (state.visible || state.notice || state.loading || state.error) ? (
        <LirieText variant="description">
          {"Consultation après la mission. Le transport reste terminé."}
        </LirieText>
      ) : null}
      {state.showHaltPrompt ? (
        <View style={styles.block}>
          <LirieText variant="description">
            {
              "Uniquement à l'arrêt. Cette confirmation n'est pas une attestation, ni une position, et ne déclenche aucun paiement."
            }
          </LirieText>
          <LirieButton
            variant="secondary"
            accessibilityLabel="Je suis à l'arrêt à destination"
            label="Je suis à l'arrêt à destination"
            onPress={onConfirmHalt}
          />
        </View>
      ) : null}
      {state.showSettlementAction ? (
        <LirieButton
          variant="secondary"
          accessibilityLabel="Règlement du transport"
          label="Règlement du transport"
          onPress={onOpen}
        />
      ) : null}
      {state.loading ? <LirieText variant="message">{"Lecture du règlement…"}</LirieText> : null}
      {state.visible && state.contract ? (
        <View style={styles.block}>
          <LirieText variant="subtitle">{"SIMULATION — AUCUN PAIEMENT RÉEL"}</LirieText>
          <DriverSettlementPanel contract={state.contract} showBack onBack={onClose} />
        </View>
      ) : null}
      {state.notice ? (
        <View style={styles.block}>
          <LirieText variant="message">{state.notice}</LirieText>
          <LirieButton
            variant="quiet"
            accessibilityLabel="Retour à la mission"
            label="Retour à la mission"
            onPress={onClose}
          />
        </View>
      ) : null}
      {state.offline || state.error ? (
        <View style={styles.block}>
          <LirieText variant="message">
            {state.offline
              ? "Connexion indisponible. Aucun paiement n'est tenté. Aucune autorisation n'est conservée."
              : "Réponse du serveur indisponible. Aucune autorisation n'est conservée."}
          </LirieText>
          <LirieButton
            variant="quiet"
            accessibilityLabel="Retour à la mission"
            label="Retour à la mission"
            onPress={onClose}
          />
        </View>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    width: "100%",
    marginTop: lirieSpace.lg,
    padding: lirieSpace.md,
    gap: lirieSpace.sm,
    backgroundColor: lirieColor.white,
    borderColor: lirieColor.border,
    borderWidth: 1,
    borderRadius: lirieRadius.card,
  },
  block: { gap: lirieSpace.sm },
});
