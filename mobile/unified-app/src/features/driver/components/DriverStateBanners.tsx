import { useEffect, useMemo, useState } from "react";
import { AppState, Linking, Platform, Pressable, StyleSheet, Text, View } from "react-native";
import { semanticDanger, semanticWarning } from "../../../design/responsive/colors";
import NetInfo from "@react-native-community/netinfo";
import * as Location from "expo-location";
import { useSession } from "../../../core/sessionProvider";
import {
  getPushPermissionDenied,
  subscribePushPermissionDenied,
} from "../../../core/notifications/pushPermissionState";
import { getTrackingSnapshot } from "../tracking";
import { useSocketStatus } from "../hooks";
import { FONT_SIZE } from "../../../design/responsive/typographyTokens";
import {
  checkBatteryOptimizationStatus,
  getOemBatteryGuidance,
  openOemBatterySettings,
  requestIgnoreBatteryOptimizations,
} from "../services/batteryOptimization";

/** Suivi mission dégradé (permission/FGS) : alerte transitoire via DriverTrackingBannerHost dans _layout. */

const MAX_FONT_MULTIPLIER = 1.35;

/** Alerte compacte : une ligne courte, titre en semi-gras + corps. */
function Banner(props: {
  title: string;
  message: string;
  tone?: "warn" | "error";
  actionLabel?: string;
  onAction?: () => void;
}) {
  const tokens = props.tone === "error" ? semanticDanger : semanticWarning;
  return (
    <Pressable
      style={[styles.banner, { backgroundColor: tokens.bg }]}
      accessibilityRole="alert"
      accessibilityLiveRegion="polite"
      onPress={props.onAction}
      disabled={!props.onAction}
    >
      <Text
        maxFontSizeMultiplier={MAX_FONT_MULTIPLIER}
        style={[styles.alertText, { color: tokens.fg }]}
      >
        <Text style={styles.alertLead}>{props.title}</Text>
        {props.message ? ` ${props.message}` : null}
        {props.actionLabel ? (
          <Text style={styles.alertAction}> {props.actionLabel}</Text>
        ) : null}
      </Text>
    </Pressable>
  );
}

export function DriverStateBanners() {
  const { status } = useSession();
  const socketStatus = useSocketStatus();
  const [isOffline, setIsOffline] = useState(false);
  const [gpsEnabled, setGpsEnabled] = useState(true);
  const [trackingDepth, setTrackingDepth] = useState(0);
  const [pushPermissionDenied, setPushPermissionDeniedState] = useState(getPushPermissionDenied());
  const [batteryOptimizationActive, setBatteryOptimizationActive] = useState(false);
  const [oemGuidance, setOemGuidance] = useState(() => getOemBatteryGuidance());

  useEffect(() => {
    return subscribePushPermissionDenied(() => {
      setPushPermissionDeniedState(getPushPermissionDenied());
    });
  }, []);

  useEffect(() => {
    if (Platform.OS !== "android") return;
    let mounted = true;

    const refresh = async () => {
      const result = await checkBatteryOptimizationStatus();
      if (!mounted) return;
      setBatteryOptimizationActive(result.checked && result.isIgnoring === false);
    };

    void refresh();
    const sub = AppState.addEventListener("change", (next) => {
      if (next === "active") void refresh();
    });

    return () => {
      mounted = false;
      sub.remove();
    };
  }, []);

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
    pushPermissionDenied ||
    isOffline ||
    (socketStatus.degraded && socketStatus.connected) ||
    !gpsEnabled ||
    batteryOptimizationActive ||
    status === "error" ||
    showTrackingWarning;

  if (!hasBanner) return null;

  return (
    <View style={styles.stack}>
      {pushPermissionDenied ? (
        <Banner
          title="Notifications desactivees"
          message="Activez les notifications pour recevoir vos missions."
          tone="error"
          actionLabel="Ouvrir les reglages"
          onAction={() => {
            void Linking.openSettings();
          }}
        />
      ) : null}
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
      {batteryOptimizationActive ? (
        <Banner
          title="Optimisation batterie active"
          message="Vos positions GPS peuvent ne pas etre transmises en arriere-plan."
          tone="warn"
          actionLabel="Appuyer ici pour corriger"
          onAction={() => {
            void requestIgnoreBatteryOptimizations().then(async () => {
              const result = await checkBatteryOptimizationStatus();
              setBatteryOptimizationActive(result.checked && result.isIgnoring === false);
            });
          }}
        />
      ) : null}
      {oemGuidance.hasOemSettings && batteryOptimizationActive ? (
        <Banner
          title="Reglages fabricant requis"
          message={`Sur ${oemGuidance.manufacturer || "votre appareil"}, ouvrez aussi Auto-start / apps protegees.`}
          tone="warn"
          actionLabel="Reglages avances fabricant"
          onAction={() => {
            void openOemBatterySettings();
          }}
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
  alertAction: {
    fontWeight: "600",
    textDecorationLine: "underline",
  },
});
