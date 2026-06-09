import { Platform, StyleSheet, View } from "react-native";
import * as Location from "expo-location";

import { useDriverBackgroundTrackingUi } from "../hooks/useDriverBackgroundTrackingUi";
import { useTransientTrackingBanner } from "../hooks/useTransientTrackingBanner";
import { DriverTrackingBanner } from "./DriverTrackingBanner";

export function DriverTrackingBannerHost() {
  const trackingUi = useDriverBackgroundTrackingUi();
  const visible = useTransientTrackingBanner(trackingUi.showBanner, trackingUi.bannerKind);

  const handleRequestBgPermission = async () => {
    await Location.requestBackgroundPermissionsAsync().catch(() => undefined);
  };

  if (!visible) return null;

  return (
    <View style={styles.host} pointerEvents="box-none">
      <DriverTrackingBanner
        ui={{ ...trackingUi, showBanner: true }}
        onRequestPermission={handleRequestBgPermission}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  host: {
    position: "absolute",
    top: 0,
    left: 0,
    right: 0,
    zIndex: 50,
    paddingHorizontal: 12,
    paddingTop: Platform.OS === "ios" ? 4 : 6,
    paddingBottom: 4,
  },
});
