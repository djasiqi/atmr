import { useCallback, useEffect, useState } from "react";
import { Linking, Platform, StyleSheet, View } from "react-native";
import * as Location from "expo-location";

import { isFeatureEnabled } from "../../../core/featureFlags/registry";
import { AppText } from "../../../design/ui/AppText";
import { useTrackingWindowState } from "../services/trackingWindow";
import {
  getDriverAvailabilityActive,
  subscribeDriverAvailability,
} from "../services/driverAvailabilityBridge";
import {
  getDriverTrackingBridgeSnapshot,
  refreshDriverTrackingBridgeState,
  subscribeDriverTrackingBridge,
} from "../services/driverTrackingBridge";
import {
  isPresenceDisclosureAccepted,
  isPresenceDisclosureDeclined,
  markPresenceDisclosureAccepted,
  markPresenceDisclosureDeclined,
} from "../services/liveTrackingDisclosureSession";
import { PresenceAvailabilityDisclosureModal } from "./PresenceAvailabilityDisclosureModal";

/**
 * Affiche la disclosure « disponibilité flotte » avant tout FGS présence.
 * Évite un blocage silencieux côté bridge sans UI.
 */
export function DriverPresenceDisclosureHost() {
  const window = useTrackingWindowState();
  const workWindowEnabled = isFeatureEnabled("driver_tracking_work_window_enabled");
  const [driverAvailable, setDriverAvailable] = useState(() => getDriverAvailabilityActive());
  const [bridgeSnapshot, setBridgeSnapshot] = useState(getDriverTrackingBridgeSnapshot());
  const [disclosureVisible, setDisclosureVisible] = useState(false);
  const [disclosurePending, setDisclosurePending] = useState(false);
  const [showOpenSettings, setShowOpenSettings] = useState(false);

  useEffect(() => subscribeDriverAvailability(() => {
    setDriverAvailable(getDriverAvailabilityActive());
  }), []);

  useEffect(() => {
    return subscribeDriverTrackingBridge(setBridgeSnapshot);
  }, []);

  const presenceWindowWanted =
    driverAvailable && workWindowEnabled && window.isOpen && bridgeSnapshot.missionId == null;

  const needsDisclosure =
    presenceWindowWanted && !isPresenceDisclosureAccepted() && !isPresenceDisclosureDeclined();

  const showDeclinedHint =
    presenceWindowWanted &&
    !isPresenceDisclosureAccepted() &&
    isPresenceDisclosureDeclined();

  useEffect(() => {
    if (needsDisclosure) {
      setDisclosureVisible(true);
      setShowOpenSettings(false);
    }
  }, [needsDisclosure]);

  const handleDisclosureContinue = useCallback(async () => {
    setDisclosurePending(true);
    markPresenceDisclosureAccepted();
    const fg = await Location.requestForegroundPermissionsAsync().catch(() => ({ granted: false }));
    if (!fg.granted) {
      setDisclosurePending(false);
      setShowOpenSettings(true);
      return;
    }
    const bg = await Location.requestBackgroundPermissionsAsync().catch(() => ({ granted: false }));
    if (!bg.granted) {
      setDisclosurePending(false);
      setShowOpenSettings(true);
      return;
    }
    setDisclosurePending(false);
    setDisclosureVisible(false);
    setShowOpenSettings(false);
    refreshDriverTrackingBridgeState();
  }, []);

  const handleDisclosureCancel = useCallback(() => {
    markPresenceDisclosureDeclined();
    setDisclosureVisible(false);
    setDisclosurePending(false);
    setShowOpenSettings(false);
  }, []);

  if (!presenceWindowWanted && !disclosureVisible) {
    return null;
  }

  return (
    <View style={styles.host} pointerEvents="box-none">
      {showDeclinedHint ? (
        <View style={styles.hintCard} pointerEvents="none">
          <AppText variant="caption" style={styles.hintText}>
            Disponibilité flotte : la localisation n&apos;est pas active. Acceptez la disclosure
            pour être visible sur la carte dispatch.
          </AppText>
        </View>
      ) : null}
      <PresenceAvailabilityDisclosureModal
        visible={disclosureVisible}
        pending={disclosurePending}
        showOpenSettings={showOpenSettings}
        onCancel={handleDisclosureCancel}
        onContinue={() => void handleDisclosureContinue()}
        onOpenSettings={() => void Linking.openSettings()}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  host: {
    position: "absolute",
    top: Platform.OS === "ios" ? 56 : 48,
    left: 0,
    right: 0,
    zIndex: 49,
    paddingHorizontal: 12,
    gap: 8,
  },
  hintCard: {
    backgroundColor: "rgba(255, 248, 220, 0.96)",
    borderRadius: 10,
    borderWidth: 1,
    borderColor: "rgba(180, 120, 0, 0.25)",
    paddingHorizontal: 12,
    paddingVertical: 8,
  },
  hintText: {
    color: "#5C4A00",
    lineHeight: 17,
  },
});
