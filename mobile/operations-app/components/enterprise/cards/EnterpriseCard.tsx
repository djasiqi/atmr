import React from "react";
import { View, StyleSheet, StyleProp, ViewStyle } from "react-native";

type Props = {
  children: React.ReactNode;
  style?: StyleProp<ViewStyle>;
  bleed?: boolean;
};

// ✅ Palette professionnelle cohérente avec le dashboard driver
const palette = {
  background: "#FFFFFF",
  border: "rgba(15,54,43,0.08)",
  shadow: "rgba(15,54,43,0.08)",
};

export const EnterpriseCard: React.FC<Props> = ({ children, style, bleed }) => {
  return (
    <View style={[styles.card, bleed && styles.cardBleed, style]}>{children}</View>
  );
};

const styles = StyleSheet.create({
  card: {
    backgroundColor: palette.background,
    borderRadius: 20,
    padding: 18,
    borderWidth: 1,
    borderColor: palette.border,
    shadowColor: palette.shadow,
    shadowOpacity: 1,
    shadowOffset: { width: 0, height: 4 },
    shadowRadius: 12,
    elevation: 2,
  },
  cardBleed: {
    paddingHorizontal: 0,
  },
});

export default EnterpriseCard;
