import { useEffect, useMemo, useState } from "react";
import { StyleSheet, Text, View } from "react-native";
import { semanticDanger, semanticWarning } from "../../../design/responsive/colors";
import NetInfo from "@react-native-community/netinfo";
import * as Location from "expo-location";
import { useSession } from "../../../core/sessionProvider";
import { getTrackingSnapshot } from "../tracking";
import { useSocketStatus } from "../hooks";
import { FONT_SIZE } from "../../../design/responsive/typographyTokens";

const MAX_FONT_MULTIPLIER = 1.35;

/** Alerte compacte : une ligne courte, titre en semi-gras + corps. */
function Banner(props: { title: string; message: string; tone?: "warn" | "error" }) {
  const tokens = props.tone === "error" ? semanticDanger : semanticWarning;
  return (
    <View
      style={[styles.banner, { backgroundColor: tokens.bg }]}
      accessibilityRole="alert"
      accessibilityLiveRegion="polite"
    >
      <Text
        maxFontSizeMultiplier={MAX_FONT_MULTIPLIER}
        style={[styles.alertText, { color: tokens.fg }]}
      >
        <Text style={styles.alertLead}>{props.title}</Text>
        {props.message ? ` ${props.message}` : null}
      </Text>
    </View>
  );
}

export function DriverStateBanners() {
  const { status } = useSession();
  const socketStatus = useSocketStatus();
  const [isOffline, setIsOffline] = useState(false);
  const [gpsEnabled, setGpsEnabled] = useState(true);
  const [trackingDepth, setTrackingDepth] = useState(0);

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
    };
    void tick();
    const interval = setInterval(() => void tick(), 10_000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  const showTrackingWarning = useMemo(() => trackingDepth > 0, [trackingDepth]);

  const hasBanner =
    isOffline ||
    (socketStatus.degraded && socketStatus.connected) ||
    !gpsEnabled ||
    status === "error" ||
    showTrackingWarning;

  if (!hasBanner) return null;

  return (
    <View style={styles.stack}>
      {isOffline ? (
        <Banner
          title="Mode hors ligne"
          message="Connexion indisponible. Les actions sont mises en file et rejouees."
          tone="warn"
        />
      ) : null}
      {socketStatus.degraded && socketStatus.connected ? (
        <Banner
          title="Temps reel instable"
          message="Sync degradee, leger retard possible."
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
    </View>
  );
}

const styles = StyleSheet.create({
  stack: {
    gap: 6,
  },
  banner: {
    borderRadius: 6,
    paddingVertical: 6,
    paddingHorizontal: 10,
  },
  alertText: {
    fontSize: FONT_SIZE.px12,
    lineHeight: 16,
    fontWeight: "400",
  },
  alertLead: {
    fontWeight: "600",
  },
});
