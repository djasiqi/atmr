import React, { useMemo } from "react";
import {
  View,
  TouchableOpacity,
  Text,
  StyleSheet,
  ViewStyle,
} from "react-native";

import {
  useEnterpriseContext,
  EnterpriseMode,
} from "@/context/EnterpriseContext";

const OPTIONS: Array<{ value: EnterpriseMode; label: string }> = [
  { value: "manual", label: "Manuel" },
  { value: "semi_auto", label: "Semi-auto" },
  { value: "fully_auto", label: "Auto" },
];

type ModeSwitchProps = {
  value?: EnterpriseMode;
  onChange?: (mode: EnterpriseMode) => void;
  condensed?: boolean;
  style?: ViewStyle;
};

export const ModeSwitch: React.FC<ModeSwitchProps> = ({
  value,
  onChange,
  condensed = false,
  style,
}) => {
  const ctx = useEnterpriseContext();
  const current = value ?? ctx.mode;
  const handleChange = onChange ?? ctx.setMode;

  const containerStyle = useMemo(
    () => [styles.container, condensed && styles.containerCondensed, style],
    [condensed, style]
  );

  return (
    <View style={containerStyle}>
      {OPTIONS.map((option) => {
        const active = option.value === current;
        return (
          <TouchableOpacity
            key={option.value}
            style={[styles.pill, active && styles.pillActive]}
            onPress={() => handleChange(option.value)}
            activeOpacity={0.85}
          >
            <Text style={[styles.pillLabel, active && styles.pillLabelActive]}>
              {option.label}
            </Text>
          </TouchableOpacity>
        );
      })}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: "row",
    backgroundColor: "rgba(15,25,55,0.75)",
    borderRadius: 18,
    padding: 6,
    gap: 6,
  },
  containerCondensed: {
    paddingVertical: 4,
  },
  pill: {
    flex: 1,
    borderRadius: 14,
    paddingVertical: 10,
    alignItems: "center",
    justifyContent: "center",
  },
  pillActive: {
    backgroundColor: "rgba(255,255,255,0.18)",
  },
  pillLabel: {
    color: "rgba(222,230,255,0.65)",
    fontWeight: "600",
    fontSize: 14,
  },
  pillLabelActive: {
    color: "#FFFFFF",
  },
});

export default ModeSwitch;
