/**
 * Panneau pédagogique tracking — informatif, non bloquant pour la navigation.
 *
 * - onboarding : premier contact
 * - needs_attention : réglages incomplets ou révoqués (sans reset onboarding)
 *
 * NB : la lecture des prérequis device (permissions Location, GPS) est
 * faite directement via expo-location, indépendamment du feature flag
 * `tracking_background_enabled`. Le flag contrôle l'orchestration runtime
 * du tracking BG, pas l'état réel des permissions OS.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ActivityIndicator,
  AppState,
  Linking,
  Platform,
  Pressable,
  StyleSheet,
  Text,
  View,
} from "react-native";
import * as Location from "expo-location";

import { semanticDanger, semanticSuccess, semanticWarning } from "../../../design/responsive/colors";
import { markNotificationDisclosureAccepted } from "../../../core/notifications/notificationDisclosurePersistence";
import {
  checkBatteryOptimizationStatus,
  getOemBatteryGuidance,
  openOemBatterySettings,
  requestIgnoreBatteryOptimizations,
} from "../services/batteryOptimization";
import { markLiveTrackingDisclosureAccepted } from "../services/liveTrackingDisclosureSession";
import {
  markTrackingOnboarded,
  setTrackingNeedsAttention,
} from "../services/trackingReadinessPersistence";
import { MissionLiveTrackingDisclosureModal } from "./MissionLiveTrackingDisclosureModal";
import { NotificationPermissionDisclosure } from "./NotificationPermissionDisclosure";

export type TrackingReadinessSnapshot = {
  ready: boolean;
  bgPermissionGranted: boolean;
  fgPermissionGranted: boolean;
  notificationsGranted: boolean;
  batteryExempt: boolean;
  gpsEnabled: boolean;
  oem: string | null;
  hasOemSettings: boolean;
};

async function readNotificationsGranted(): Promise<boolean> {
  if (Platform.OS === "web") return true;
  try {
    const Notifications = await import("expo-notifications");
    const perm = await Notifications.getPermissionsAsync();
    return Boolean(perm.granted || perm.status === "granted");
  } catch {
    return false;
  }
}

export async function evaluateTrackingReadiness(): Promise<TrackingReadinessSnapshot> {
  const [fg, bg, servicesEnabled, battery, notificationsGranted, oem] = await Promise.all([
    Location.getForegroundPermissionsAsync().catch(() => ({ granted: false })),
    Location.getBackgroundPermissionsAsync().catch(() => ({ granted: false })),
    Location.hasServicesEnabledAsync().catch(() => false),
    checkBatteryOptimizationStatus(),
    readNotificationsGranted(),
    Promise.resolve(getOemBatteryGuidance()),
  ]);

  const fgPermissionGranted = Boolean(fg.granted);
  const bgPermissionGranted = Boolean(bg.granted);
  const gpsEnabled = Boolean(servicesEnabled);
  const batteryExempt =
    Platform.OS !== "android" || !battery.checked || battery.isIgnoring !== false;

  const ready =
    gpsEnabled &&
    bgPermissionGranted &&
    fgPermissionGranted &&
    notificationsGranted &&
    batteryExempt;

  return {
    ready,
    bgPermissionGranted,
    fgPermissionGranted,
    notificationsGranted,
    batteryExempt,
    gpsEnabled,
    oem: oem.oem,
    hasOemSettings: oem.hasOemSettings,
  };
}

type Props = {
  onReadyChange?: (ready: boolean) => void;
  /** Si true, le composant ne s'affiche jamais (utile une fois l'onboarding fait). */
  silent?: boolean;
  mode?: "onboarding" | "needs_attention";
};

export function DriverTrackingReadinessGate(props: Props) {
  const { onReadyChange, silent } = props;
  const [loading, setLoading] = useState(true);
  const [snapshot, setSnapshot] = useState<TrackingReadinessSnapshot | null>(null);
  const [bgDisclosureVisible, setBgDisclosureVisible] = useState(false);
  const [bgDisclosurePending, setBgDisclosurePending] = useState(false);
  const [notifDisclosureVisible, setNotifDisclosureVisible] = useState(false);
  const [notifDisclosurePending, setNotifDisclosurePending] = useState(false);
  const onboardedRef = useRef(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    const next = await evaluateTrackingReadiness();
    setSnapshot(next);
    onReadyChange?.(next.ready);
    if (next.ready) {
      if (!onboardedRef.current) {
        onboardedRef.current = true;
        void markTrackingOnboarded().catch(() => undefined);
      }
      void setTrackingNeedsAttention(false).catch(() => undefined);
    } else {
      void setTrackingNeedsAttention(true).catch(() => undefined);
    }
    setLoading(false);
  }, [onReadyChange]);

  useEffect(() => {
    void refresh();
    const sub = AppState.addEventListener("change", (next) => {
      if (next === "active") void refresh();
    });
    return () => sub.remove();
  }, [refresh]);

  const requestBgWithDisclosure = useCallback(async () => {
    setBgDisclosureVisible(true);
  }, []);

  const handleBgDisclosureContinue = useCallback(async () => {
    setBgDisclosurePending(true);
    markLiveTrackingDisclosureAccepted();
    const fg = await Location.requestForegroundPermissionsAsync().catch(() => ({ granted: false }));
    if (!fg.granted) {
      setBgDisclosurePending(false);
      setBgDisclosureVisible(false);
      await refresh();
      return;
    }
    await Location.requestBackgroundPermissionsAsync().catch(() => undefined);
    setBgDisclosurePending(false);
    setBgDisclosureVisible(false);
    await refresh();
  }, [refresh]);

  const handleNotifDisclosureAccept = useCallback(async () => {
    setNotifDisclosurePending(true);
    await markNotificationDisclosureAccepted();
    setNotifDisclosurePending(false);
    setNotifDisclosureVisible(false);
    await refresh();
  }, [refresh]);

  const checklist = useMemo(() => {
    if (!snapshot) return [];
    return [
      {
        ok: snapshot.fgPermissionGranted,
        label: "Localisation autorisée",
      },
      {
        ok: snapshot.bgPermissionGranted,
        label: "Localisation arrière-plan (Toujours)",
      },
      {
        ok: snapshot.notificationsGranted,
        label: "Notifications autorisées",
      },
      {
        ok: snapshot.batteryExempt,
        label: "Optimisation batterie désactivée",
      },
      {
        ok: snapshot.gpsEnabled,
        label: "GPS activé",
      },
    ];
  }, [snapshot]);

  if (silent) return null;

  if (loading && !snapshot) {
    return (
      <View style={styles.container}>
        <ActivityIndicator />
      </View>
    );
  }

  if (snapshot?.ready) {
    return null;
  }

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Préparation tracking obligatoire</Text>
      <Text style={styles.subtitle}>
        Avant votre première mission, vérifiez les réglages ci-dessous.
      </Text>
      {checklist.map((item) => (
        <Text
          key={item.label}
          style={[styles.checkItem, { color: item.ok ? semanticSuccess.fg : semanticDanger.fg }]}
        >
          {item.ok ? "✓" : "✗"} {item.label}
        </Text>
      ))}
      {snapshot?.hasOemSettings ? (
        <Text style={styles.oemHint}>
          Fabricant détecté ({snapshot.oem}). Ouvrez aussi les réglages avancés du fabricant
          (auto-start / apps protégées).
        </Text>
      ) : null}
      <View style={styles.actions}>
        <Pressable style={styles.button} onPress={() => void requestBgWithDisclosure()}>
          <Text style={styles.buttonText}>Autoriser localisation</Text>
        </Pressable>
        {!snapshot?.notificationsGranted ? (
          <Pressable style={styles.button} onPress={() => setNotifDisclosureVisible(true)}>
            <Text style={styles.buttonText}>Autoriser notifications</Text>
          </Pressable>
        ) : null}
        <Pressable style={styles.button} onPress={() => void requestIgnoreBatteryOptimizations()}>
          <Text style={styles.buttonText}>Exemption batterie</Text>
        </Pressable>
        {snapshot?.hasOemSettings ? (
          <Pressable style={styles.button} onPress={() => void openOemBatterySettings()}>
            <Text style={styles.buttonText}>Réglages fabricant</Text>
          </Pressable>
        ) : null}
        <Pressable style={styles.buttonSecondary} onPress={() => Linking.openSettings()}>
          <Text style={styles.buttonSecondaryText}>Ouvrir réglages</Text>
        </Pressable>
        <Pressable style={styles.buttonSecondary} onPress={() => void refresh()}>
          <Text style={styles.buttonSecondaryText}>Revérifier</Text>
        </Pressable>
      </View>
      <View style={[styles.warningBox, { backgroundColor: semanticWarning.bg }]}>
        <Text style={{ color: semanticWarning.fg }}>
          Le démarrage d&apos;une mission suivie (écran verrouillé) requiert ces réglages.
          Vous pouvez continuer à consulter vos missions.
        </Text>
      </View>

      <MissionLiveTrackingDisclosureModal
        visible={bgDisclosureVisible}
        pending={bgDisclosurePending}
        showOpenSettings={false}
        onCancel={() => {
          setBgDisclosureVisible(false);
          setBgDisclosurePending(false);
        }}
        onContinue={() => void handleBgDisclosureContinue()}
        onOpenSettings={() => void Linking.openSettings()}
      />
      <NotificationPermissionDisclosure
        visible={notifDisclosureVisible}
        pending={notifDisclosurePending}
        onCancel={() => {
          setNotifDisclosureVisible(false);
          setNotifDisclosurePending(false);
        }}
        onAccept={() => void handleNotifDisclosureAccept()}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    padding: 16,
    gap: 10,
  },
  title: {
    fontSize: 18,
    fontWeight: "700",
  },
  subtitle: {
    fontSize: 14,
    opacity: 0.8,
  },
  checkItem: {
    fontSize: 14,
    fontWeight: "600",
  },
  oemHint: {
    fontSize: 13,
    opacity: 0.85,
  },
  actions: {
    gap: 8,
    marginTop: 8,
  },
  button: {
    backgroundColor: "#0A7F59",
    borderRadius: 10,
    paddingVertical: 12,
    alignItems: "center",
  },
  buttonText: {
    color: "#fff",
    fontWeight: "700",
  },
  buttonSecondary: {
    borderRadius: 10,
    paddingVertical: 10,
    alignItems: "center",
    borderWidth: 1,
    borderColor: "#CBD5E1",
  },
  buttonSecondaryText: {
    fontWeight: "600",
  },
  warningBox: {
    borderRadius: 10,
    padding: 12,
    marginTop: 8,
  },
  readyBox: {
    backgroundColor: semanticSuccess.bg,
    borderRadius: 12,
  },
  readyTitle: {
    color: semanticSuccess.fg,
    fontWeight: "700",
    fontSize: 16,
  },
  readyBody: {
    color: semanticSuccess.fg,
  },
});
