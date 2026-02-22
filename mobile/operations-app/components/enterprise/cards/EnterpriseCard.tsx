import React from "react";
import { View, StyleSheet, StyleProp, ViewStyle } from "react-native";
import { createShadow } from "@/styles/shadowStyles";

type Props = {
  children: React.ReactNode;
  style?: StyleProp<ViewStyle>;
  bleed?: boolean;
};

const palette = {
  background: "#FFFFFF",
  border: "rgba(0,121,107,0.08)",
};

export const EnterpriseCard: React.FC<Props> = ({ children, style, bleed }) => {
  return (
    <View style={[styles.card, bleed && styles.cardBleed, style]}>{children}</View>
  );
};

const styles = StyleSheet.create({
  card: {
    backgroundColor: palette.background,
    borderRadius: 14,
    padding: 14,
    borderWidth: 1,
    borderColor: palette.border,
    ...createShadow({
      shadowColor: "#000",
      shadowOpacity: 0.04,
      shadowOffset: { width: 0, height: 2 },
      shadowRadius: 8,
      elevation: 2,
    }),
  },
  cardBleed: {
    paddingHorizontal: 0,
  },
});

export default EnterpriseCard;
