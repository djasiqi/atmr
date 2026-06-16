import { StyleSheet, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { AppButton, Modal } from "../../../design/responsive";
import { AppText } from "../../../design/ui/AppText";
import { E } from "../../company/theme/enterpriseOpsTheme";

type ReleaseConfirmationModalProps = {
  visible: boolean;
  missionId: number | null;
  pending: boolean;
  onCancel: () => void;
  onConfirm: () => void;
};

/**
 * Confirmation de libération de mission (parité `operations-app/MissionCard.tsx`
 * lignes 1070+). Sémantique distincte d'« Annuler » : le chauffeur rend la
 * mission au pool dispatch — réassignée à un autre chauffeur, sans facturation.
 *
 * Côté API : transition `CANCELLED` avec `cancel_reason: "RELEASE"`
 * (cf. `useDriverStatusTransition` → `api.ts`).
 */
export function ReleaseConfirmationModal(props: ReleaseConfirmationModalProps) {
  const missionLabel = `#${props.missionId ?? "n/a"}`;

  return (
    <Modal
      visible={props.visible}
      title="Libérer la course"
      subtitle={`Mission ${missionLabel} · opération dispatch`}
      onClose={props.onCancel}
      presentation="bottomSheet"
      sheetBodyMaxHeightRatio={0.56}
      footer={
        <View style={styles.footerWrap}>
          <AppText variant="caption" style={styles.footerHint}>
            Action sans facturation client
          </AppText>
          <View style={styles.footerRow}>
            <AppButton
              title="Annuler"
              variant="secondary"
              onPress={props.onCancel}
              disabled={props.pending}
              style={styles.footerButtonSecondary}
            />
            <AppButton
              title={props.pending ? "Libération..." : "Confirmer la libération"}
              variant="primary"
              onPress={props.onConfirm}
              disabled={props.pending}
              style={styles.footerButtonPrimary}
            />
          </View>
        </View>
      }
    >
      <View style={styles.body}>
        <View style={styles.heroRow}>
          <View style={styles.iconPill}>
            <Ionicons name="return-up-back-outline" size={18} color={E.BRAND} />
          </View>
          <View style={styles.heroContent}>
            <AppText variant="body" style={styles.heroTitle}>
              Mission {missionLabel}
            </AppText>
            <AppText variant="caption" style={styles.heroSubtitle}>
              Réassignation immédiate au pool dispatch
            </AppText>
          </View>
        </View>

        <View style={styles.warningCard}>
          <Ionicons name="alert-circle-outline" size={17} color={E.BRAND} />
          <AppText variant="body" style={styles.warningText}>
            Cette action retire la mission de votre file active. Un autre chauffeur pourra la reprendre.
          </AppText>
        </View>

        <View style={styles.infoCard}>
          <Ionicons name="information-circle-outline" size={16} color={E.BRAND} />
          <AppText variant="caption" style={styles.infoText}>
            Sans facturation : opération dispatch, pas une annulation client.
          </AppText>
        </View>
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  body: {
    gap: 12,
  },
  heroRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
    paddingVertical: 4,
  },
  iconPill: {
    width: 34,
    height: 34,
    borderRadius: 17,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "rgba(0, 121, 107, 0.08)",
    borderWidth: 1,
    borderColor: "rgba(0, 121, 107, 0.16)",
  },
  heroContent: {
    flex: 1,
    gap: 2,
  },
  heroTitle: {
    color: E.TEXT_MAIN,
    fontWeight: "700",
  },
  heroSubtitle: {
    color: E.TEXT_SEC,
    lineHeight: 17,
  },
  warningCard: {
    flexDirection: "row",
    alignItems: "flex-start",
    gap: 8,
    paddingVertical: 10,
    paddingHorizontal: 12,
    backgroundColor: "rgba(0, 121, 107, 0.06)",
    borderRadius: 10,
    borderWidth: 1,
    borderColor: "rgba(0, 121, 107, 0.14)",
  },
  warningText: {
    flex: 1,
    color: E.TEXT_MAIN,
    lineHeight: 20,
  },
  infoCard: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    paddingVertical: 9,
    paddingHorizontal: 12,
    backgroundColor: "rgba(0, 121, 107, 0.04)",
    borderRadius: 10,
    borderWidth: 1,
    borderColor: "rgba(0, 121, 107, 0.1)",
  },
  infoText: {
    color: E.TEXT_SEC,
    flex: 1,
    lineHeight: 17,
  },
  footerRow: {
    flexDirection: "row",
    gap: 8,
  },
  footerWrap: {
    gap: 10,
  },
  footerHint: {
    color: E.TEXT_SEC,
  },
  footerButtonSecondary: {
    flex: 1,
    minHeight: 46,
    borderRadius: 11,
    borderColor: "rgba(0, 121, 107, 0.32)",
  },
  footerButtonPrimary: {
    flex: 1,
    minHeight: 46,
    borderRadius: 11,
  },
});
