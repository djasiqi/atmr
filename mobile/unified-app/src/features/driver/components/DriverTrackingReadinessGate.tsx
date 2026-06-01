/**
 * Gate bloquant avant première mission — vérifie les prérequis tracking.
 */
import { useCallback, useEffect, useMemo, useState } from "react";
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
import {
  checkBatteryOptimizationStatus,
  getOemBatteryGuidance,
  openOemBatterySettings,
  requestIgnoreBatteryOptimizations,
} from "../services/batteryOptimization";
import { evaluateBackgroundTrackingGate } from "../services/backgroundTrackingGating";

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
  const [gate, battery, notificationsGranted, oem] = await Promise.all([
    evaluateBackgroundTrackingGate(),
    checkBatteryOptimizationStatus(),
    readNotificationsGranted(),
    Promise.resolve(getOemBatteryGuidance()),
  ]);

  const fg = await Location.getForegroundPermissionsAsync().catch(() => ({ granted: false }));
  const bgPermissionGranted = gate.permission === "granted";
  const fgPermissionGranted = Boolean(fg.granted);
  const batteryExempt =
    Platform.OS !== "android" || !battery.checked || battery.isIgnoring !== false;

  const ready =
    gate.servicesEnabled &&
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
    gpsEnabled: gate.servicesEnabled,
    oem: oem.oem,
    hasOemSettings: oem.hasOemSettings,
  };
}

type Props = {
  onReadyChange?: (ready: boolean) => void;
};

export function DriverTrackingReadinessGate(props: Props) {
  const [loading, setLoading] = useState(true);
  const [snapshot, setSnapshot] = useState<TrackingReadinessSnapshot | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    const next = await evaluateTrackingReadiness();
    setSnapshot(next);
    props.onReadyChange?.(next.ready);
    setLoading(false);
  }, [props.onReadyChange]);

  useEffect(() => {
    void refresh();
    const sub = AppState.addEventListener("change", (next) => {
      if (next === "active") void refresh();
    });
    return () => sub.remove();
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

  if (loading && !snapshot) {
    return (
      <View style={styles.container}>
        <ActivityIndicator />
      </View>
    );
  }

  if (snapshot?.ready) {
    return (
      <View style={[styles.container, styles.readyBox]}>
        <Text style={styles.readyTitle}>Tracking prêt</Text>
        <Text style={styles.readyBody}>Vous pouvez démarrer une mission.</Text>
      </View>
    );
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
        <Pressable style={styles.button} onPress={() => void Location.requestBackgroundPermissionsAsync()}>
          <Text style={styles.buttonText}>Autoriser localisation</Text>
        </Pressable>
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
          La première mission reste bloquée tant que tous les prérequis ne sont pas validés.
        </Text>
      </View>
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
