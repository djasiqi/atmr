import { Platform, StyleSheet, View } from "react-native";
import {
  LIRIE_GOOGLE_ATTRIBUTION_ANDROID_LEFT_BLEED_PX,
  LIRIE_GOOGLE_ATTRIBUTION_ANDROID_LEFT_PATCH_WIDTH_PX,
  LIRIE_GOOGLE_MAP_ATTRIBUTION_MASK_COLOR,
  nativeGoogleAttributionMaskHeight,
} from "./lirieMapChrome";

type Props = {
  /** Couleur du bandeau (alignée fond carte / carte UI). */
  backgroundColor?: string;
};

/**
 * Masque la mention « Google » en bas des cartes `react-native-maps` (Android / iOS).
 * Complète `LirieMapLogoClip` pour le coin bas-gauche Android.
 */
export function LirieNativeMapAttributionMask({
  backgroundColor = LIRIE_GOOGLE_MAP_ATTRIBUTION_MASK_COLOR,
}: Props) {
  if (Platform.OS === "web") return null;

  const maskHeight = nativeGoogleAttributionMaskHeight();

  return (
    <>
      <View
        pointerEvents="none"
        accessibilityElementsHidden
        style={[styles.bottomBand, { height: maskHeight, backgroundColor }]}
      />
      {Platform.OS === "android" ? (
        <View
          pointerEvents="none"
          accessibilityElementsHidden
          style={[
            styles.androidLeftPatch,
            {
              width: LIRIE_GOOGLE_ATTRIBUTION_ANDROID_LEFT_PATCH_WIDTH_PX,
              height: maskHeight + LIRIE_GOOGLE_ATTRIBUTION_ANDROID_LEFT_BLEED_PX,
              backgroundColor,
            },
          ]}
        />
      ) : null}
    </>
  );
}

const styles = StyleSheet.create({
  bottomBand: {
    position: "absolute",
    left: 0,
    right: 0,
    bottom: 0,
    zIndex: 2,
  },
  androidLeftPatch: {
    position: "absolute",
    left: 0,
    bottom: 0,
    zIndex: 3,
  },
});
