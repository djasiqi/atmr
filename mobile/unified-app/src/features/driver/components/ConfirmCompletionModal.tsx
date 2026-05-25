import { StyleSheet, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { AppButton, Modal } from "../../../design/responsive";
import { AppText } from "../../../design/ui/AppText";
import { E } from "../../company/theme/enterpriseOpsTheme";

type ConfirmCompletionModalProps = {
  visible: boolean;
  missionId: number | null;
  /** Nom client affiché dans l'en-tête (ex. « Catherine BRONNIMANN »). */
  clientLabel?: string | null;
  pending: boolean;
  onCancel: () => void;
  onConfirm: () => void;
};

export function ConfirmCompletionModal({
  visible,
  missionId,
  clientLabel,
  pending,
  onCancel,
  onConfirm,
}: ConfirmCompletionModalProps) {
  const missionLabel = `#${missionId ?? "n/a"}`;
  const heroTitle = clientLabel?.trim() || `Mission ${missionLabel}`;

  return (
    <Modal
      visible={visible}
      title="Terminer la mission"
      subtitle={`Mission ${missionLabel} · clôture de course`}
      onClose={onCancel}
      presentation="bottomSheet"
      sheetBodyMaxHeightRatio={0.56}
      footer={
        <View style={styles.footerWrap}>
          <AppText variant="caption" style={styles.footerHint}>
            Patient déposé à destination
          </AppText>
          <View style={styles.footerRow}>
            <AppButton
              title="Annuler"
              variant="secondary"
              onPress={onCancel}
              disabled={pending}
              style={styles.footerButtonSecondary}
            />
            <AppButton
              title={pending ? "Clôture..." : "Confirmer la fin"}
              variant="primary"
              onPress={onConfirm}
              disabled={pending}
              style={styles.footerButtonPrimary}
            />
          </View>
        </View>
      }
    >
      <View style={styles.body}>
        <View style={styles.heroRow}>
          <View style={styles.iconPill}>
            <Ionicons name="checkmark-done-outline" size={18} color={E.BRAND} />
          </View>
          <View style={styles.heroContent}>
            <AppText variant="body" style={styles.heroTitle}>
              {heroTitle}
            </AppText>
            <AppText variant="caption" style={styles.heroSubtitle}>
              Valider la fin du transport et libérer la mission
            </AppText>
          </View>
        </View>

        <View style={styles.warningCard}>
          <Ionicons name="alert-circle-outline" size={17} color={E.BRAND} />
          <AppText variant="body" style={styles.warningText}>
            Cette action clôt définitivement la course. Vous ne pourrez plus modifier le statut ni reprendre la mission.
          </AppText>
        </View>

        <View style={styles.infoCard}>
          <Ionicons name="information-circle-outline" size={16} color={E.BRAND} />
          <AppText variant="caption" style={styles.infoText}>
            À confirmer uniquement une fois le patient déposé au point d&apos;arrivée prévu.
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
