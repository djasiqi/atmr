import { StyleSheet, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { AppButton, Modal } from "../../../design/responsive";
import { AppText } from "../../../design/ui/AppText";
import { E } from "../../company/theme/enterpriseOpsTheme";

type Props = {
  visible: boolean;
  pending: boolean;
  onCancel: () => void;
  onConfirm: () => void;
};

export function UnavailableConfirmationModal({
  visible,
  pending,
  onCancel,
  onConfirm,
}: Props) {
  return (
    <Modal
      visible={visible}
      title="Passer indisponible"
      subtitle="Disponibilité chauffeur · dispatch"
      onClose={onCancel}
      presentation="bottomSheet"
      sheetBodyMaxHeightRatio={0.58}
      footer={
        <View style={styles.footerWrap}>
          <AppText variant="caption" style={styles.footerHint}>
            Réactivable à tout moment depuis le dashboard
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
              title={pending ? "Mise à jour…" : "Confirmer indisponible"}
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
            <Ionicons name="pause-circle-outline" size={18} color={E.BRAND} />
          </View>
          <View style={styles.heroContent}>
            <AppText variant="body" style={styles.heroTitle}>
              Vous ne recevrez plus de courses
            </AppText>
            <AppText variant="caption" style={styles.heroSubtitle}>
              Le dispatch ne pourra plus vous assigner de nouvelles missions
            </AppText>
          </View>
        </View>

        <View style={styles.warningCard}>
          <Ionicons name="alert-circle-outline" size={17} color={E.BRAND} />
          <AppText variant="body" style={styles.warningText}>
            En mode indisponible, votre position GPS n&apos;est plus mise à jour pour la
            répartition des courses (hors mission déjà en cours).
          </AppText>
        </View>

        <View style={styles.infoCard}>
          <Ionicons name="information-circle-outline" size={16} color={E.BRAND} />
          <AppText variant="caption" style={styles.infoText}>
            Repassez « Disponible » pour réapparaître sur la carte dispatch et recevoir de
            nouvelles assignations.
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
    color: E.TEXT,
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
    color: E.TEXT,
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
