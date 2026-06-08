import { Platform, StyleSheet, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { AppButton, Modal } from "../../../design/responsive";
import { AppText } from "../../../design/ui/AppText";
import { E } from "../../company/theme/enterpriseOpsTheme";

type Props = {
  visible: boolean;
  pending: boolean;
  showOpenSettings: boolean;
  onCancel: () => void;
  onContinue: () => void;
  onOpenSettings: () => void;
};

export function MissionLiveTrackingDisclosureModal({
  visible,
  pending,
  showOpenSettings,
  onCancel,
  onContinue,
  onOpenSettings,
}: Props) {
  return (
    <Modal
      visible={visible}
      title="Suivi de mission"
      subtitle="Localisation · mission en cours"
      onClose={onCancel}
      presentation="bottomSheet"
      sheetBodyMaxHeightRatio={0.62}
      footer={
        <View style={styles.footerWrap}>
          <View style={styles.footerRow}>
            <AppButton
              title="Annuler"
              variant="secondary"
              onPress={onCancel}
              disabled={pending}
              style={styles.footerButtonSecondary}
            />
            {showOpenSettings ? (
              <AppButton
                title="Ouvrir les réglages"
                variant="primary"
                onPress={onOpenSettings}
                disabled={pending}
                style={styles.footerButtonPrimary}
              />
            ) : (
              <AppButton
                title={pending ? "Vérification…" : "Continuer"}
                variant="primary"
                onPress={onContinue}
                disabled={pending}
                style={styles.footerButtonPrimary}
              />
            )}
          </View>
        </View>
      }
    >
      <View style={styles.body}>
        <View style={styles.heroRow}>
          <View style={styles.iconPill}>
            <Ionicons name="navigate-outline" size={18} color={E.BRAND} />
          </View>
          <View style={styles.heroContent}>
            <AppText variant="body" style={styles.heroTitle}>
              Suivi lorsque l&apos;écran est verrouillé
            </AppText>
          </View>
        </View>

        <AppText variant="body" style={styles.bodyText}>
          Pour permettre le suivi de la mission lorsque l&apos;écran est verrouillé,
          l&apos;autorisation de localisation en arrière-plan est requise. Cette
          localisation est utilisée uniquement pour le suivi opérationnel de la mission
          en cours.
        </AppText>
        <AppText variant="caption" style={styles.bodyMuted}>
          Vous pouvez consulter vos missions sans cette autorisation.
        </AppText>

        {Platform.OS === "android" ? (
          <View style={styles.infoCard}>
            <Ionicons name="information-circle-outline" size={16} color={E.BRAND} />
            <AppText variant="caption" style={styles.infoText}>
              Pendant la mission, une notification persistante indique que la
              localisation est active (« Mission en cours — localisation active »).
            </AppText>
          </View>
        ) : null}
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
  },
  heroTitle: {
    color: E.TEXT,
    fontWeight: "700",
  },
  bodyText: {
    color: E.TEXT,
    lineHeight: 20,
  },
  bodyMuted: {
    color: E.TEXT_SEC,
    lineHeight: 18,
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
