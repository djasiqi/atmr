import { useCallback, useEffect, useState } from "react";
import { StyleSheet, View } from "react-native";

import { isFeatureEnabled } from "../../../core/featureFlags/registry";
import {
  ensureNotificationDisclosureSyncedWithOsPermission,
  markNotificationDisclosureAccepted,
  readNotificationDisclosureAccepted,
  subscribeNotificationDisclosureAccepted,
} from "../../../core/notifications/notificationDisclosurePersistence";
import {
  getDisclosureShowRequestCount,
  subscribePushRegistrationState,
} from "../../../core/notifications/pushRegistrationState";
import { useSession } from "../../../core/sessionProvider";
import { NotificationPermissionDisclosure } from "./NotificationPermissionDisclosure";

/**
 * Disclosure notifications indépendante du tracking BG.
 * N'appelle jamais requestPermissionsAsync — conformité Play Store.
 */
export function DriverNotificationDisclosureHost() {
  const { status, activeContext } = useSession();
  const pushEnabled = isFeatureEnabled("driver_push_enabled");
  const isDriverContext =
    status === "ready" && activeContext?.context_type === "driver";

  const [disclosureAccepted, setDisclosureAccepted] = useState<boolean | null>(null);
  const [disclosureVisible, setDisclosureVisible] = useState(false);
  const [disclosurePending, setDisclosurePending] = useState(false);

  const refreshDisclosureState = useCallback(async () => {
    await ensureNotificationDisclosureSyncedWithOsPermission();
    const accepted = await readNotificationDisclosureAccepted();
    setDisclosureAccepted(accepted);
    if (!accepted && pushEnabled && isDriverContext) {
      setDisclosureVisible(true);
    } else {
      setDisclosureVisible(false);
    }
  }, [pushEnabled, isDriverContext]);

  useEffect(() => {
    void refreshDisclosureState();
  }, [refreshDisclosureState]);

  useEffect(() => {
    return subscribeNotificationDisclosureAccepted(() => {
      void refreshDisclosureState();
    });
  }, [refreshDisclosureState]);

  useEffect(() => {
    return subscribePushRegistrationState(() => {
      if (getDisclosureShowRequestCount() > 0) {
        setDisclosureVisible(true);
      }
    });
  }, []);

  const handleAccept = useCallback(async () => {
    setDisclosurePending(true);
    try {
      await markNotificationDisclosureAccepted();
      setDisclosureVisible(false);
    } finally {
      setDisclosurePending(false);
    }
  }, []);

  const handleCancel = useCallback(() => {
    setDisclosureVisible(false);
  }, []);

  if (!pushEnabled || !isDriverContext || disclosureAccepted === true) {
    return null;
  }

  return (
    <View style={styles.host} pointerEvents="box-none">
      <NotificationPermissionDisclosure
        visible={disclosureVisible}
        pending={disclosurePending}
        onCancel={handleCancel}
        onAccept={() => void handleAccept()}
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
    zIndex: 50,
    justifyContent: "flex-end",
  },
});
