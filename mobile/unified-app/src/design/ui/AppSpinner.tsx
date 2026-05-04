import { ActivityIndicator, View, type StyleProp, type ViewStyle } from "react-native";
import { brandPrimary } from "../responsive/colors";
import { AppText } from "./AppText";

export type AppSpinnerProps = {
  size?: "small" | "large";
  color?: string;
  /** Libellé sous l’indicateur (accessibilité / contexte). */
  label?: string;
  style?: StyleProp<ViewStyle>;
  accessibilityLabel?: string;
};

export function AppSpinner({
  size = "large",
  color = brandPrimary,
  label,
  style,
  accessibilityLabel,
}: AppSpinnerProps) {
  return (
    <View style={[{ alignItems: "center", justifyContent: "center", gap: 8 }, style]}>
      <ActivityIndicator
        size={size}
        color={color}
        accessibilityLabel={accessibilityLabel ?? label}
      />
      {label ? (
        <AppText variant="caption" style={{ textAlign: "center" }}>
          {label}
        </AppText>
      ) : null}
    </View>
  );
}
