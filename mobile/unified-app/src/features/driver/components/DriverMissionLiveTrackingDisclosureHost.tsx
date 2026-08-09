import { useCallback, useEffect, useState } from "react";
import { StyleSheet, View } from "react-native";

import {
  cancelMissionLiveTrackingDisclosure,
  continueMissionLiveTrackingDisclosure,
  getMissionLiveTrackingDisclosureSnapshot,
  openMissionLiveTrackingSettings,
  subscribeMissionLiveTrackingDisclosure,
} from "../services/missionLiveTrackingDisclosureBridge";
import { MissionLiveTrackingDisclosureModal } from "./MissionLiveTrackingDisclosureModal";

/** Modale mission centralisée dans le layout chauffeur (P2). */
export function DriverMissionLiveTrackingDisclosureHost() {
  const [snapshot, setSnapshot] = useState(getMissionLiveTrackingDisclosureSnapshot());

  useEffect(() => {
    return subscribeMissionLiveTrackingDisclosure(() => {
      setSnapshot(getMissionLiveTrackingDisclosureSnapshot());
    });
  }, []);

  const handleCancel = useCallback(() => {
    cancelMissionLiveTrackingDisclosure();
  }, []);

  const handleContinue = useCallback(() => {
    continueMissionLiveTrackingDisclosure();
  }, []);

  const handleOpenSettings = useCallback(() => {
    openMissionLiveTrackingSettings();
  }, []);

  return (
    <View style={styles.host} pointerEvents="box-none">
      <MissionLiveTrackingDisclosureModal
        visible={snapshot.visible}
        pending={snapshot.pending}
        showOpenSettings={snapshot.showOpenSettings}
        compact={snapshot.compact}
        onCancel={handleCancel}
        onContinue={handleContinue}
        onOpenSettings={handleOpenSettings}
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
    bottom: 0,
    zIndex: 51,
    justifyContent: "flex-end",
  },
});
