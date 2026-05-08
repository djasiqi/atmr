import { useEffect, useMemo, useState } from "react";
import { StyleSheet, View } from "react-native";
import { AppText } from "../../../design/ui/AppText";
import { semanticDanger, semanticWarning } from "../../../design/responsive/colors";
import NetInfo from "@react-native-community/netinfo";
import * as Location from "expo-location";
import { useSession } from "../../../core/sessionProvider";
import { getTrackingSnapshot } from "../tracking";
import { useSocketStatus, useTrackingState } from "../hooks";
import { driverOfflineQueue } from "../offlineQueue";

function Banner(props: { title: string; message: string; tone?: "warn" | "error" }) {
  const tokens = props.tone === "error" ? semanticDanger : semanticWarning;
  return (
    <View
      style={[styles.banner, { borderColor: tokens.border, backgroundColor: tokens.bg }]}
      accessibilityRole="text"
    >
      <AppText variant="sectionTitle" style={{ color: tokens.fg }}>
        {props.title}
      </AppText>
      <AppText variant="body" style={{ color: tokens.fg }}>
        {props.message}
      </AppText>
    </View>
  );
}

export function DriverStateBanners() {
  const { status } = useSession();
  const socketStatus = useSocketStatus();
  const trackingState = useTrackingState();
  const [isOffline, setIsOffline] = useState(false);
  const [gpsEnabled, setGpsEnabled] = useState(true);
  const [trackingDepth, setTrackingDepth] = useState(0);
  const [transitionQueueCount, setTransitionQueueCount] = useState(0);
  const [transitionQueueOldestAgeMs, setTransitionQueueOldestAgeMs] = useState(0);

  useEffect(() => {
    const unsubscribe = NetInfo.addEventListener((state) => {
      const connected = Boolean(state.isConnected) && state.isInternetReachable !== false;
      setIsOffline(!connected);
    });
    return unsubscribe;
  }, []);

  useEffect(() => {
    let mounted = true;
    const tick = async () => {
      try {
        const enabled = await Location.hasServicesEnabledAsync();
        if (mounted) setGpsEnabled(enabled);
      } catch {
        if (mounted) setGpsEnabled(false);
      }
      if (mounted) {
        setTrackingDepth(getTrackingSnapshot().queueDepth ?? 0);
      }
      try {
        const transitionSnapshot = await driverOfflineQueue.getSnapshot();
        if (mounted) {
          setTransitionQueueCount(transitionSnapshot.queuedCount);
          setTransitionQueueOldestAgeMs(transitionSnapshot.oldestAgeMs);
        }
      } catch {
        if (mounted) {
          setTransitionQueueCount(0);
          setTransitionQueueOldestAgeMs(0);
        }
      }
    };
    void tick();
    const interval = setInterval(() => void tick(), 10_000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  const showTrackingWarning = useMemo(() => trackingDepth > 0, [trackingDepth]);
  const showTransitionPendingWarning = useMemo(() => transitionQueueCount > 0, [transitionQueueCount]);
  const transitionExpiredRisk = transitionQueueOldestAgeMs >= 10 * 60 * 1000;

  return (
    <View style={styles.stack}>
      {isOffline ? (
        <Banner
          title="Mode hors ligne"
          message="Connexion indisponible. Les actions sont mises en file et rejouees."
          tone="warn"
        />
      ) : null}
      {!socketStatus.connected ? (
        <Banner
          title="Connexion temps reel indisponible"
          message={
            socketStatus.authExhausted
              ? "Votre session live a expire. Reconnectez-vous pour reprendre les mises a jour mission."
              : socketStatus.reconnecting
              ? "Reconnexion en cours. Les mises a jour live peuvent etre retardees."
              : "Le canal live est indisponible. Le mode polling reste actif."
          }
          tone={socketStatus.authExhausted ? "error" : "warn"}
        />
      ) : null}
      {socketStatus.degraded && socketStatus.connected ? (
        <Banner
          title="Connexion temps reel instable"
          message="Le service poursuit la synchronisation en mode degrade. Les mises a jour peuvent etre legerement differees."
          tone="warn"
        />
      ) : null}
      {!gpsEnabled ? (
        <Banner
          title="GPS desactive"
          message="Activez la localisation pour maintenir le suivi mission."
          tone="error"
        />
      ) : null}
      {status === "error" ? (
        <Banner
          title="Session indisponible"
          message="La session a expire ou le bootstrap a echoue. Reconnectez-vous."
          tone="error"
        />
      ) : null}
      {showTrackingWarning ? (
        <Banner
          title="Synchronisation en cours"
          message={`Position en attente d'envoi (${trackingDepth}).`}
          tone="warn"
        />
      ) : null}
      {socketStatus.degraded ? (
        <Banner
          title="Connexion GPS instable"
          message="Le suivi continue en mode degrade. Certaines positions peuvent etre rejouees avec retard."
          tone="warn"
        />
      ) : null}
      {showTransitionPendingWarning ? (
        <Banner
          title="Action mission en attente"
          message={
            transitionExpiredRisk
              ? "Une action mission n'a pas encore ete envoyee. Verification requise."
              : `Des actions mission restent en attente d'envoi (${transitionQueueCount}).`
          }
          tone={transitionExpiredRisk ? "error" : "warn"}
        />
      ) : null}
      {!trackingState.isTracking && trackingState.mode === "idle" && status === "ready" ? (
        <Banner
          title="Tracking inactif"
          message="Le suivi chauffeur est inactif. Vérifiez qu'une mission est bien engagée."
          tone="warn"
        />
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  /** Même rythme vertical que le dashboard entreprise (`gap: 14`). */
  stack: {
    gap: 14,
  },
  /** Cartes alerte : rayon 16 comme les tuiles KPI / sections. */
  banner: {
    borderWidth: 1,
    borderRadius: 16,
    paddingVertical: 14,
    paddingHorizontal: 16,
    gap: 6,
  },
});
