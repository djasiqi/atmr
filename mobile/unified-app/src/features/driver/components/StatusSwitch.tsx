import { StyleSheet, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { AppText } from "../../../design/ui/AppText";
import { AppButton } from "../../../design/responsive";
import { createShadow } from "../../../styles/shadowStyles";
import { E } from "../../company/theme/enterpriseOpsTheme";
import type { DriverMissionStatus, DriverTransitionStatus } from "../types";
import { getDriverStatusUx } from "../statusDictionary";

type Props = {
  mode: "availability" | "mission";
  isAvailable?: boolean;
  onToggleAvailability?: () => void;
  missionStatus?: DriverMissionStatus | null;
  onTransition?: (target: DriverTransitionStatus) => void;
  disabled?: boolean;
  pending?: boolean;
};

const C = {
  text: E.TEXT,
  textSub: E.TEXT_SEC,
  textMuted: E.TEXT_MUTED,
  border: E.BORDER,
  cardBg: E.CARD,
  brand: E.BRAND,
  brandSoft: "rgba(0, 121, 107, 0.08)",
  off: "#94A3B8",
} as const;

const cardShadow = createShadow({
  shadowColor: "#000000",
  shadowOffset: { width: 0, height: 2 },
  shadowOpacity: 0.04,
  shadowRadius: 8,
  elevation: 2,
});

export function StatusSwitch(props: Props) {
  if (props.mode === "availability") {
    const available = Boolean(props.isAvailable);
    const dotColor = available ? C.brand : C.off;
    return (
      <View style={styles.card}>
        <View style={styles.headerRow}>
          <View style={styles.iconWrap} accessibilityElementsHidden>
            <Ionicons name="pulse-outline" size={16} color={C.brand} />
          </View>
          <AppText variant="sectionTitle" style={styles.title}>
            Disponibilité chauffeur
          </AppText>
        </View>

        <View style={styles.statusRow}>
          <View style={[styles.statusDot, { backgroundColor: dotColor }]} accessibilityElementsHidden />
          <AppText variant="label" style={styles.statusValue}>
            {available ? "Disponible" : "Indisponible"}
          </AppText>
        </View>

        <AppButton
          title={
            props.pending
              ? "Mise à jour…"
              : available
                ? "Passer indisponible"
                : "Passer disponible"
          }
          variant="primary"
          onPress={() => props.onToggleAvailability?.()}
          disabled={props.disabled || props.pending}
        />
      </View>
    );
  }

  const ux = getDriverStatusUx(props.missionStatus ?? null);
  return (
    <View style={styles.card}>
      <View style={styles.headerRow}>
        <View style={styles.iconWrap} accessibilityElementsHidden>
          <Ionicons name="git-branch-outline" size={16} color={C.brand} />
        </View>
        <AppText variant="sectionTitle" style={styles.title}>
          Transitions mission
        </AppText>
      </View>
      <AppText variant="body" style={styles.statusValue}>
        État courant : {ux.label}
      </AppText>
      {ux.nextTransitions.length === 0 ? (
        <AppText variant="bodyMuted">Aucune transition disponible.</AppText>
      ) : (
        ux.nextTransitions.map((target) => (
          <AppButton
            key={target}
            title={props.pending ? "Enregistrement…" : `Passer ${target}`}
            variant="secondary"
            onPress={() => props.onTransition?.(target)}
            disabled={props.disabled || props.pending}
          />
        ))
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  /** Réf. `dashboardSection` company : carte blanche, rayon 16, padding 16, ombre + bordure. */
  card: {
    alignSelf: "stretch",
    backgroundColor: C.cardBg,
    borderWidth: 1,
    borderColor: C.border,
    borderRadius: 16,
    padding: 16,
    gap: 12,
    ...cardShadow,
  },
  headerRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
  },
  iconWrap: {
    width: 28,
    height: 28,
    borderRadius: 8,
    backgroundColor: C.brandSoft,
    alignItems: "center",
    justifyContent: "center",
  },
  title: {
    color: C.text,
    fontSize: 16,
    fontWeight: "700",
    flex: 1,
    minWidth: 0,
  },
  statusRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
  },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  statusValue: {
    color: C.text,
    fontWeight: "600",
  },
});
