import { useCallback, useState } from "react";
import { Linking, Platform, StyleSheet, View } from "react-native";
import * as Location from "expo-location";

import { useDriverBackgroundTrackingUi } from "../hooks/useDriverBackgroundTrackingUi";
import { useTransientTrackingBanner } from "../hooks/useTransientTrackingBanner";
import { markLiveTrackingDisclosureAccepted } from "../services/liveTrackingDisclosureSession";
import { notifyMissionTrackingCapabilityRefresh } from "../services/missionLiveTrackingEligibility";
import { DriverTrackingBanner } from "./DriverTrackingBanner";
import { MissionLiveTrackingDisclosureModal } from "./MissionLiveTrackingDisclosureModal";

export function DriverTrackingBannerHost() {
  const trackingUi = useDriverBackgroundTrackingUi();
  const visible = useTransientTrackingBanner(trackingUi.showBanner, trackingUi.bannerKind);
  const [disclosureVisible, setDisclosureVisible] = useState(false);
  const [disclosurePending, setDisclosurePending] = useState(false);

  const handleRequestBgPermission = useCallback(() => {
    setDisclosureVisible(true);
  }, []);

  const handleDisclosureContinue = useCallback(async () => {
    setDisclosurePending(true);
    markLiveTrackingDisclosureAccepted();
    const fg = await Location.requestForegroundPermissionsAsync().catch(() => ({ granted: false }));
    if (!fg.granted) {
      setDisclosurePending(false);
      setDisclosureVisible(false);
      return;
    }
    await Location.requestBackgroundPermissionsAsync().catch(() => undefined);
    setDisclosurePending(false);
    setDisclosureVisible(false);
    notifyMissionTrackingCapabilityRefresh();
  }, []);

  if (!visible) return null;

  return (
    <View style={styles.host} pointerEvents="box-none">
      <DriverTrackingBanner
        ui={{ ...trackingUi, showBanner: true }}
        onRequestPermission={handleRequestBgPermission}
      />
      <MissionLiveTrackingDisclosureModal
        visible={disclosureVisible}
        pending={disclosurePending}
        showOpenSettings={false}
        onCancel={() => {
          setDisclosureVisible(false);
          setDisclosurePending(false);
        }}
        onContinue={() => void handleDisclosureContinue()}
        onOpenSettings={() => void Linking.openSettings()}
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
