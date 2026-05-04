import type { ReactNode } from "react";
import { StyleSheet, View, type StyleProp, type ViewStyle } from "react-native";
import { useAppViewport } from "./useAppViewport";

export type ResponsiveContainerProps = {
  children: ReactNode;
  style?: StyleProp<ViewStyle>;
};

/** Colonne centrée, largeur max = contentWidth (usableWidth − gutters, plafonnée). */
export function ResponsiveContainer({ children, style }: ResponsiveContainerProps) {
  const { contentWidth } = useAppViewport();

  return (
    <View style={[styles.root, { maxWidth: contentWidth }, style]}>{children}</View>
  );
}

const styles = StyleSheet.create({
  root: {
    width: "100%",
    alignSelf: "center",
  },
});
