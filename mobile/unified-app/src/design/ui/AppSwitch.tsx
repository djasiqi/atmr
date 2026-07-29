import type { ReactNode } from "react";
import { Pressable, Switch, View, type StyleProp, type ViewStyle } from "react-native";
import { brandPrimary } from "../responsive/colors";
import { useResponsiveTokens } from "../responsive/useResponsiveTokens";

export type AppSwitchProps = {
  value: boolean;
  onValueChange: (next: boolean) => void;
  /** Contenu à droite du switch (souvent `AppText`). */
  label: ReactNode;
  /** Style du conteneur ligne (flexDirection row). */
  style?: StyleProp<ViewStyle>;
  accessibilityLabel?: string;
  disabled?: boolean;
};

export function AppSwitch({
  value,
  onValueChange,
  label,
  style,
  accessibilityLabel,
  disabled = false,
}: AppSwitchProps) {
  const t = useResponsiveTokens();
  const trackOff = "#E6EEEB";
  const trackOn = brandPrimary;

  return (
    <View style={[{ flexDirection: "row", alignItems: "center", gap: t.spacingSm }, style]}>
      <Switch
        value={value}
        onValueChange={onValueChange}
        disabled={disabled}
        trackColor={{ false: trackOff, true: trackOn }}
        thumbColor="#FFFFFF"
        ios_backgroundColor={trackOff}
        accessibilityRole="switch"
        accessibilityState={{ checked: value }}
        accessibilityLabel={accessibilityLabel}
      />
      <Pressable
        onPress={() => {
          if (disabled) return;
          onValueChange(!value);
        }}
        accessibilityRole="none"
        importantForAccessibility="no-hide-descendants"
        style={{ flex: 1, minWidth: 0, flexShrink: 1 }}
      >
        {label}
      </Pressable>
    </View>
  );
}
