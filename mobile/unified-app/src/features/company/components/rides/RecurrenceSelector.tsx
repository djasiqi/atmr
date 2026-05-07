import { Pressable, StyleSheet, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { useResponsiveTokens } from "../../../../design/responsive";
import { AppText } from "../../../../design/ui/AppText";
import { E } from "../../theme/enterpriseOpsTheme";

export type RecurrenceFrequency = "daily" | "weekly" | "custom";

type RecurrenceSelectorProps = {
  value: RecurrenceFrequency;
  onChange: (value: RecurrenceFrequency) => void;
  /** Faux : masque le libellé (ligne déjà titrée par le parent). */
  showLabel?: boolean;
};

const RECURRENCE_LABEL: Record<RecurrenceFrequency, string> = {
  daily: "Quotidien",
  weekly: "Hebdomadaire",
  custom: "Perso",
};

const ICONS: Record<RecurrenceFrequency, keyof typeof Ionicons.glyphMap> = {
  daily: "today-outline",
  weekly: "calendar-outline",
  custom: "git-branch-outline",
};

const styles = StyleSheet.create({
  wrap: {
    gap: 8,
  },
  segmented: {
    padding: 3,
    borderRadius: 11,
    backgroundColor: "rgba(0, 121, 107, 0.09)",
    borderWidth: 1,
    borderColor: "rgba(0, 121, 107, 0.14)",
  },
  segmentedRow: {
    flexDirection: "row" as const,
    flexWrap: "nowrap" as const,
    alignItems: "stretch" as const,
    gap: 3,
    width: "100%" as const,
  },
  segment: {
    flexGrow: 1,
    flexShrink: 1,
    flexBasis: 0,
    minWidth: 0,
    flexDirection: "row" as const,
    alignItems: "center" as const,
    justifyContent: "center" as const,
    gap: 4,
    paddingVertical: 8,
    paddingHorizontal: 7,
    borderRadius: 9,
    minHeight: 42,
    borderWidth: StyleSheet.hairlineWidth,
  },
  segmentOn: {
    backgroundColor: "#FFFFFF",
    borderColor: "rgba(0, 121, 107, 0.22)",
    shadowColor: "rgba(15, 23, 42, 0.12)",
    shadowOpacity: 1,
    shadowRadius: 8,
    shadowOffset: { width: 0, height: 2 },
    elevation: 2,
  },
  segmentOff: {
    backgroundColor: "transparent",
    borderColor: "transparent",
  },
  labelOn: {
    color: E.BRAND,
    fontWeight: "700" as const,
    fontSize: 12,
    lineHeight: 15,
    letterSpacing: 0.15,
    flexShrink: 1,
  },
  labelOff: {
    color: E.TEXT_SEC,
    fontWeight: "600" as const,
    fontSize: 12,
    lineHeight: 15,
    flexShrink: 1,
  },
});

export function RecurrenceSelector({ value, onChange, showLabel = true }: RecurrenceSelectorProps) {
  const t = useResponsiveTokens();
  const options = ["daily", "weekly", "custom"] as const;
  return (
    <View style={[styles.wrap, { gap: t.fieldGap }]}>
      {showLabel ? (
        <AppText variant="label" style={{ marginBottom: 2 }}>
          Fréquence
        </AppText>
      ) : null}
      <View style={styles.segmented}>
        <View style={styles.segmentedRow}>
          {options.map((option) => {
            const on = value === option;
            return (
              <Pressable
                key={option}
                onPress={() => onChange(option)}
                style={[styles.segment, on ? styles.segmentOn : styles.segmentOff]}
                accessibilityRole="button"
                accessibilityLabel={
                  option === "custom"
                    ? "Personnalisé : jours de la semaine au choix"
                    : RECURRENCE_LABEL[option]
                }
                accessibilityState={{ selected: on }}
              >
                <Ionicons name={ICONS[option]} size={17} color={on ? E.BRAND : E.TEXT_MUTED} />
                <AppText
                  numberOfLines={1}
                  ellipsizeMode="tail"
                  style={on ? styles.labelOn : styles.labelOff}
                >
                  {RECURRENCE_LABEL[option]}
                </AppText>
              </Pressable>
            );
          })}
        </View>
      </View>
    </View>
  );
}
