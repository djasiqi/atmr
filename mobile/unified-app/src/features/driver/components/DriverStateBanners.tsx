import { useEffect, useMemo, useState } from "react";
import { View } from "react-native";
import { AppText } from "../../../design/ui/AppText";
import NetInfo from "@react-native-community/netinfo";
import * as Location from "expo-location";
import { useSession } from "../../../core/sessionProvider";
import { getTrackingSnapshot } from "../tracking";
import { useSocketStatus, useTrackingState } from "../hooks";
import { driverOfflineQueue } from "../offlineQueue";

function Banner(props: { title: string; message: string; tone?: "warn" | "error" }) {
  const errorTone = props.tone === "error";
  return (
    <View
      style={{
        borderWidth: 1,
        borderColor: errorTone ? "#B00020" : "#8a6d3b",
        backgroundColor: errorTone ? "#fdecea" : "#fff8e1",
        borderRadius: 8,
        padding: 10,
        gap: 2,
      }}
    >
      {/* DS_EXCEPTION: couleurs sémantiques bannière warning / erreur sur fond teinté */}
      <AppText variant="sectionTitle" style={{ color: errorTone ? "#8a1f1f" : "#6a5320" }}>
        {props.title}
      </AppText>
      <AppText variant="body" style={{ color: errorTone ? "#8a1f1f" : "#6a5320" }}>
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
    <View style={{ gap: 8 }}>
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
          title="Synchronisation tracking"
          message={`Des points de localisation restent en attente (${trackingDepth}).`}
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
          message="Le suivi chauffeur est inactif. Verifiez qu'une mission est bien engagee."
          tone="warn"
        />
      ) : null}
    </View>
  );
}
