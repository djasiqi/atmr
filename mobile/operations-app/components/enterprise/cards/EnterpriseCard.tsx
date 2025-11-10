import React from "react";
import { View, StyleSheet, StyleProp, ViewStyle } from "react-native";

type Props = {
  children: React.ReactNode;
  style?: StyleProp<ViewStyle>;
  bleed?: boolean;
};

const palette = {
  background: "rgba(10,34,26,0.92)",
  border: "rgba(59,143,105,0.24)",
  shadow: "#04150F",
};

export const EnterpriseCard: React.FC<Props> = ({ children, style, bleed }) => {
  return (
    <View style={[styles.card, bleed && styles.cardBleed, style]}>{children}</View>
  );
};

const styles = StyleSheet.create({
  card: {
    backgroundColor: palette.background,
    borderRadius: 18,
    padding: 16,
    borderWidth: 1,
    borderColor: palette.border,
    shadowColor: palette.shadow,
    shadowOpacity: 0.25,
    shadowOffset: { width: 0, height: 4 },
    shadowRadius: 12,
    elevation: 3,
  },
  cardBleed: {
    paddingHorizontal: 0,
  },
});

export default EnterpriseCard;
