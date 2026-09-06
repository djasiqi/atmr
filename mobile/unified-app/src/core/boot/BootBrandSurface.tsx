import type { ReactNode } from "react";
import { StyleSheet, View } from "react-native";
import { BootBrandMark } from "./BootBrandMark";
import { SPLASH_BACKGROUND_COLOR } from "./bootSurface";

type Props = { children?: ReactNode };

/**
 * Surface de hold / Redirect : fond LIRIE + logo, jamais #FFFFFF.
 */
export function BootBrandSurface({ children }: Props) {
  return (
    <View style={styles.root}>
      <BootBrandMark />
      {children}
    </View>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: SPLASH_BACKGROUND_COLOR,
  },
});
