import { useCallback, useEffect, useMemo, useState } from "react";
import { Platform, StyleSheet, Text, View } from "react-native";
import { Marker } from "react-native-maps";

import {
  FLEET_CLUSTER_COUNT_BADGE_ANCHOR,
  resolveFleetClusterCountBadgeLayout,
  resolveFleetClusterBadgeFontSize,
} from "./fleetLirieClusterMarker";

type Props = {
  coordinate: { latitude: number; longitude: number };
  count: number;
  onPress?: () => void;
  opacity?: number;
};

/** Pastille compteur en Text natif (lisible) — l’icône PNG est un marqueur séparé. */
export function ClusterCountBadgeMarker({ coordinate, count, onPress, opacity = 1 }: Props) {
  const layout = useMemo(() => resolveFleetClusterCountBadgeLayout(count), [count]);
  const fontSize = useMemo(() => resolveFleetClusterBadgeFontSize(layout.label), [layout.label]);
  const [tracksViewChanges, setTracksViewChanges] = useState(Platform.OS === "android");

  useEffect(() => {
    setTracksViewChanges(true);
    const timer = setTimeout(() => setTracksViewChanges(false), 700);
    return () => clearTimeout(timer);
  }, [count, layout.width, layout.height]);

  const stopTracking = useCallback(() => {
    setTracksViewChanges(false);
  }, []);

  return (
    <Marker
      coordinate={coordinate}
      anchor={FLEET_CLUSTER_COUNT_BADGE_ANCHOR}
      tracksViewChanges={tracksViewChanges}
      zIndex={501}
      opacity={opacity}
      onPress={onPress}
    >
      <View
        style={[
          styles.badge,
          {
            minWidth: layout.width,
            height: layout.height,
            borderRadius: layout.height / 2,
            paddingHorizontal: layout.label.length > 1 ? 4 : 0,
          },
        ]}
        onLayout={stopTracking}
        collapsable={false}
        renderToHardwareTextureAndroid
      >
        <Text style={[styles.text, { fontSize, lineHeight: fontSize + 4 }]} allowFontScaling={false}>
          {layout.label}
        </Text>
      </View>
    </Marker>
  );
}

const styles = StyleSheet.create({
  badge: {
    backgroundColor: "#0F766E",
    borderWidth: 1.5,
    borderColor: "#FFFFFF",
    alignItems: "center",
    justifyContent: "center",
    ...Platform.select({
      ios: {
        shadowColor: "#0f172a",
        shadowOpacity: 0.4,
        shadowRadius: 3,
        shadowOffset: { width: 0, height: 2 },
      },
      android: { elevation: 8 },
      default: {},
    }),
  },
  text: {
    color: "#FFFFFF",
    fontWeight: "800",
    textAlign: "center",
    includeFontPadding: false,
  },
});
