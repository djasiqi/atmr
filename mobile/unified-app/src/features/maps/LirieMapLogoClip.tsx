import type { ReactNode } from "react";
import { StyleSheet, View, type StyleProp, type ViewStyle } from "react-native";
import {
  resolveNativeGoogleLogoClipPx,
  lirieMapClipCanvasStyle,
  lirieMapClipViewportStyle,
} from "./lirieMapChrome";

type Props = {
  height: number;
  children: ReactNode;
  style?: StyleProp<ViewStyle>;
  /** Cockpit plein écran : le parent est en `flex:1` au lieu d’une hauteur fixe. */
  fill?: boolean;
};

const fillViewportStyle: ViewStyle = {
  flex: 1,
  overflow: "hidden",
  width: "100%",
  position: "relative",
};

function buildFillCanvasStyle(): ViewStyle {
  const clip = resolveNativeGoogleLogoClipPx();
  return {
    position: "absolute",
    top: 0,
    left: 0,
    right: 0,
    bottom: -clip,
  };
}

/**
 * Rogne le logo / mention Google en bas des cartes `react-native-maps` (iOS / Android).
 * Pas de bandeau opaque : la carte occupe toute la hauteur visible.
 */
export function LirieMapLogoClip({ height, children, style, fill = false }: Props) {
  if (fill) {
    return (
      <View style={[fillViewportStyle, style]}>
        <View style={buildFillCanvasStyle()}>{children}</View>
      </View>
    );
  }
  return (
    <View style={[lirieMapClipViewportStyle(height), styles.viewport, style]}>
      <View style={lirieMapClipCanvasStyle(height)}>{children}</View>
    </View>
  );
}

const styles = StyleSheet.create({
  viewport: {
    position: "relative",
  },
});
