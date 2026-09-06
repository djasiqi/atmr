import { Image, StyleSheet, View } from "react-native";
import {
  BOOT_BRAND_LOGO,
  BOOT_BRAND_LOGO_HEIGHT,
  BOOT_BRAND_LOGO_WIDTH,
} from "./bootSurface";

/**
 * Logo LIRIE centré — calque branding permanent pendant le boot
 * (Lottie au 1er launch, wordmark statique ensuite).
 */
export function BootBrandMark() {
  return (
    <View style={styles.wrap} pointerEvents="none">
      <Image
        source={BOOT_BRAND_LOGO}
        style={styles.logo}
        resizeMode="contain"
        accessibilityRole="image"
        accessibilityLabel="LIRIE"
      />
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    ...StyleSheet.absoluteFillObject,
    alignItems: "center",
    justifyContent: "center",
  },
  logo: {
    width: BOOT_BRAND_LOGO_WIDTH,
    height: BOOT_BRAND_LOGO_HEIGHT,
  },
});
