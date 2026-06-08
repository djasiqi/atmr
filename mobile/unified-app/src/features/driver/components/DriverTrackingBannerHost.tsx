import { Platform, StyleSheet, View } from "react-native";
import * as Location from "expo-location";

import { useDriverBackgroundTrackingUi } from "../hooks/useDriverBackgroundTrackingUi";
import { DriverTrackingBanner } from "./DriverTrackingBanner";

export function DriverTrackingBannerHost() {
  const trackingUi = useDriverBackgroundTrackingUi();

  const handleRequestBgPermission = async () => {
    await Location.requestBackgroundPermissionsAsync().catch(() => undefined);
  };

  if (!trackingUi.showBanner) return null;

  return (
    <View style={styles.host}>
      <DriverTrackingBanner ui={trackingUi} onRequestPermission={handleRequestBgPermission} />
    </View>
  );
}

const styles = StyleSheet.create({
  host: {
    paddingHorizontal: 12,
    paddingTop: Platform.OS === "ios" ? 4 : 6,
    paddingBottom: 4,
  },
});
