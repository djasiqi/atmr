import { useCallback, useEffect, useState } from "react";
import { Linking, Platform, Pressable, StyleSheet, View } from "react-native";
import * as Location from "expo-location";

import { AppText } from "../../../design/ui/AppText";
import {
  getDriverAvailabilityActive,
  subscribeDriverAvailability,
} from "../services/driverAvailabilityBridge";
import {
  getDriverTrackingBridgeSnapshot,
  refreshDriverTrackingBridgeState,
  subscribeDriverTrackingBridge,
} from "../services/driverTrackingBridge";
import { ensureNativeTrackingWhileForeground } from "../services/backgroundLocationTask";
import { setDriverPresenceContext } from "../tracking";
import {
  getDriverDisclosureOrchestrationSnapshot,
  setDriverPresenceDisclosureUiState,
  subscribeDriverDisclosureOrchestration,
} from "../services/driverDisclosureOrchestrator";
import {
  clearPresenceDisclosureDeclined,
  isPresenceDisclosureAccepted,
  isPresenceDisclosureDeclined,
  markPresenceDisclosureAccepted,
  markPresenceDisclosureDeclined,
} from "../services/liveTrackingDisclosureSession";
import { setTrackingPermissionsReady } from "../services/trackingPermissionsReady";
import { PresenceAvailabilityDisclosureModal } from "./PresenceAvailabilityDisclosureModal";

/**
 * Affiche la disclosure « disponibilité flotte » avant le tracking présence.
 * Contrat v4 : permissionsReady = FG + BG (HOME/lock). FG seul → BLOCKED, pas PRESENCE conforme.
 */
export function DriverPresenceDisclosureHost() {
  const [driverAvailable, setDriverAvailable] = useState(() => getDriverAvailabilityActive());
  const [bridgeSnapshot, setBridgeSnapshot] = useState(getDriverTrackingBridgeSnapshot());
  const [disclosureVisible, setDisclosureVisible] = useState(false);
  const [disclosurePending, setDisclosurePending] = useState(false);
  const [showOpenSettings, setShowOpenSettings] = useState(false);
  const [orchestration, setOrchestration] = useState(getDriverDisclosureOrchestrationSnapshot());

  useEffect(() => subscribeDriverAvailability(() => {
    setDriverAvailable(getDriverAvailabilityActive());
  }), []);

  useEffect(() => {
    return subscribeDriverTrackingBridge(setBridgeSnapshot);
  }, []);

  useEffect(() => {
    return subscribeDriverDisclosureOrchestration(() => {
      setOrchestration(getDriverDisclosureOrchestrationSnapshot());
    });
  }, []);

  // A2 : en service commande la présence (plus la fenêtre horaire).
  const presenceTrackingWanted =
    driverAvailable === true && bridgeSnapshot.missionId == null;

  const orchestrationAllowsPresence =
    !orchestration.blocksPresenceDisclosure && !orchestration.missionDisclosureVisible;

  const needsDisclosure =
    presenceTrackingWanted &&
    orchestrationAllowsPresence &&
    !isPresenceDisclosureAccepted() &&
    !isPresenceDisclosureDeclined();

  const showDeclinedHint =
    presenceTrackingWanted &&
    orchestrationAllowsPresence &&
    !isPresenceDisclosureAccepted() &&
    isPresenceDisclosureDeclined() &&
    !disclosureVisible;

  useEffect(() => {
    setDriverPresenceDisclosureUiState({
      presenceHintVisible: showDeclinedHint,
      presenceModalVisible: disclosureVisible,
    });
  }, [showDeclinedHint, disclosureVisible]);

  const openDisclosureModal = useCallback(() => {
    if (!orchestrationAllowsPresence) return;
    clearPresenceDisclosureDeclined();
    setShowOpenSettings(false);
    setDisclosureVisible(true);
  }, [orchestrationAllowsPresence]);

  useEffect(() => {
    if (needsDisclosure) {
      openDisclosureModal();
    }
  }, [needsDisclosure, openDisclosureModal]);

  const handleDisclosureContinue = useCallback(async () => {
    setDisclosurePending(true);
    try {
      const fgGranted = await (async () => {
        const current = await Location.getForegroundPermissionsAsync().catch(() => null);
        if (current?.granted) return true;
        const requested = await Location.requestForegroundPermissionsAsync().catch(() => null);
        return Boolean(requested?.granted);
      })();
      if (!fgGranted) {
        setTrackingPermissionsReady(false);
        setDisclosurePending(false);
        setShowOpenSettings(true);
        return;
      }

      const bgGranted = await (async () => {
        const current = await Location.getBackgroundPermissionsAsync().catch(() => null);
        if (current?.granted) return true;
        const requested = await Location.requestBackgroundPermissionsAsync().catch(() => null);
        return Boolean(requested?.granted);
      })();

      // A3 : FG+BG requis pour permissionsReady / PRESENCE conforme.
      setTrackingPermissionsReady(fgGranted && bgGranted);
      if (!bgGranted) {
        // Disclosure consentement UI OK, mais contrat non garanti → BLOCKED.
        markPresenceDisclosureAccepted();
        setDisclosurePending(false);
        setDisclosureVisible(false);
        setShowOpenSettings(true);
        setDriverPresenceContext({
          available: driverAvailable,
          windowOpen: true,
        });
        refreshDriverTrackingBridgeState();
        return;
      }

      markPresenceDisclosureAccepted();
      setDisclosurePending(false);
      setDisclosureVisible(false);
      setShowOpenSettings(false);

      if (presenceTrackingWanted) {
        setDriverPresenceContext({
          available: driverAvailable,
          windowOpen: true,
        });
        await ensureNativeTrackingWhileForeground(
          null,
          null,
          { presenceWindow: true },
          "presence_disclosure_accept"
        );
      }
      refreshDriverTrackingBridgeState();
    } catch {
      setTrackingPermissionsReady(false);
      setDisclosurePending(false);
      setShowOpenSettings(true);
    }
  }, [presenceTrackingWanted, driverAvailable]);

  const handleDisclosureCancel = useCallback(() => {
    markPresenceDisclosureDeclined();
    setDisclosureVisible(false);
    setDisclosurePending(false);
    setShowOpenSettings(false);
  }, []);

  if (!presenceTrackingWanted && !disclosureVisible) {
    return null;
  }

  return (
    <View style={styles.host} pointerEvents="box-none">
      {showDeclinedHint ? (
        <Pressable
          onPress={openDisclosureModal}
          style={styles.hintCard}
          accessibilityRole="button"
          accessibilityLabel="Activer la disponibilité flotte et ouvrir la confirmation"
        >
          <AppText variant="caption" style={styles.hintText}>
            Disponibilité flotte : la localisation n&apos;est pas active. Appuyez ici pour
            accepter la disclosure et être visible sur la carte dispatch.
          </AppText>
        </Pressable>
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
