import { StyleSheet, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { AppButton, Modal } from "../../../design/responsive";
import { AppText } from "../../../design/ui/AppText";
import { E } from "../../company/theme/enterpriseOpsTheme";

type Props = {
  visible: boolean;
  pending: boolean;
  onCancel: () => void;
  onAccept: () => void;
};

export function NotificationPermissionDisclosure({
  visible,
  pending,
  onCancel,
  onAccept,
}: Props) {
  return (
    <Modal
      visible={visible}
      title="Notifications"
      subtitle="Alertes missions et messages"
      onClose={onCancel}
      presentation="bottomSheet"
      sheetBodyMaxHeightRatio={0.55}
      footer={
        <View style={styles.footerWrap}>
          <View style={styles.footerRow}>
            <AppButton
              title="Plus tard"
              variant="secondary"
              onPress={onCancel}
              disabled={pending}
              style={styles.footerButtonSecondary}
            />
            <AppButton
              title={pending ? "…" : "Continuer"}
              variant="primary"
              onPress={onAccept}
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
            <Ionicons name="notifications-outline" size={18} color={E.BRAND} />
          </View>
          <AppText variant="body" style={styles.heroTitle}>
            Restez informé de vos missions
          </AppText>
        </View>
        <AppText variant="body" style={styles.bodyText}>
          LIRIE envoie des notifications pour les assignations de course, les messages
          d&apos;équipe et les mises à jour de statut. Vous pouvez refuser l&apos;autorisation
          dans les réglages système ; l&apos;application reste utilisable sans notifications.
        </AppText>
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  body: { gap: 12 },
  heroRow: { flexDirection: "row", alignItems: "center", gap: 10 },
  iconPill: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: "rgba(10, 127, 89, 0.12)",
    alignItems: "center",
    justifyContent: "center",
  },
  heroTitle: { flex: 1, fontWeight: "600", color: "#163A34" },
  bodyText: { color: "#5F7369", lineHeight: 22 },
  footerWrap: { width: "100%" },
  footerRow: { flexDirection: "row", gap: 10 },
  footerButtonSecondary: { flex: 1 },
  footerButtonPrimary: { flex: 1 },
});
