import { useCallback, useEffect, useState } from "react";
import { Platform, StyleSheet, View } from "react-native";

import { useDriverBackgroundTrackingUi } from "../hooks/useDriverBackgroundTrackingUi";
import { useTransientTrackingBanner } from "../hooks/useTransientTrackingBanner";
import {
  getDriverDisclosureOrchestrationSnapshot,
  subscribeDriverDisclosureOrchestration,
} from "../services/driverDisclosureOrchestrator";
import { openMissionLiveTrackingDisclosureForBanner } from "../services/missionLiveTrackingDisclosureBridge";
import { DriverTrackingBanner } from "./DriverTrackingBanner";

export function DriverTrackingBannerHost() {
  const trackingUi = useDriverBackgroundTrackingUi();
  const wantsBanner = useTransientTrackingBanner(trackingUi.showBanner, trackingUi.bannerKind);
  const [orchestration, setOrchestration] = useState(getDriverDisclosureOrchestrationSnapshot());

  useEffect(() => {
    return subscribeDriverDisclosureOrchestration(() => {
      setOrchestration(getDriverDisclosureOrchestrationSnapshot());
    });
  }, []);

  const handleRequestBgPermission = useCallback(() => {
    openMissionLiveTrackingDisclosureForBanner();
  }, []);

  const visible = wantsBanner && !orchestration.suppressTrackingBanner;

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
    zIndex: 48,
    paddingHorizontal: 12,
    paddingTop: Platform.OS === "ios" ? 4 : 6,
    paddingBottom: 4,
  },
});
