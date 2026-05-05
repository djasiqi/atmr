import { Pressable, StyleSheet, View } from "react-native";
import { useResponsiveTokens } from "../../../../design/responsive";
import { AppText } from "../../../../design/ui/AppText";
import { E } from "../../theme/enterpriseOpsTheme";

type Recurrence = "none" | "daily" | "weekly";

type RecurrenceSelectorProps = {
  value: Recurrence;
  onChange: (value: Recurrence) => void;
  /** Faux : masque le libellé (ligne déjà titrée par le parent). */
  showLabel?: boolean;
};

const RECURRENCE_LABEL: Record<Recurrence, string> = {
  none: "Aucune",
  daily: "Quotidien",
  weekly: "Hebdomadaire",
};

const styles = StyleSheet.create({
  row: {
    flexDirection: "row" as const,
    flexWrap: "wrap" as const,
    gap: 8,
  },
  chip: {
    paddingHorizontal: 14,
    paddingVertical: 10,
    borderRadius: 12,
    borderWidth: StyleSheet.hairlineWidth,
    minHeight: 46,
    justifyContent: "center" as const,
  },
  chipOn: {
    backgroundColor: E.BRAND,
    borderColor: E.BRAND,
  },
  chipOff: {
    backgroundColor: "#FFFFFF",
    borderColor: "rgba(0, 121, 107, 0.35)",
  },
  textOn: {
    color: "#FFFFFF",
    fontWeight: "700" as const,
    fontSize: 14,
  },
  textOff: {
    color: E.BRAND,
    fontWeight: "600" as const,
    fontSize: 14,
  },
});

export function RecurrenceSelector({ value, onChange, showLabel = true }: RecurrenceSelectorProps) {
  const t = useResponsiveTokens();
  const options: Recurrence[] = ["daily", "weekly"];
  return (
    <View style={{ gap: t.fieldGap }}>
      {showLabel ? <AppText variant="label">Récurrence</AppText> : null}
      <View style={styles.row}>
        {options.map((option) => {
          const on = value === option;
          return (
            <Pressable
              key={option}
              onPress={() => onChange(option)}
              style={[styles.chip, on ? styles.chipOn : styles.chipOff]}
              accessibilityRole="button"
              accessibilityState={{ selected: on }}
            >
              <AppText variant="body" style={on ? styles.textOn : styles.textOff}>
                {RECURRENCE_LABEL[option]}
              </AppText>
            </Pressable>
          );
        })}
      </View>
    </View>
  );
}
