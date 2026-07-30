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
  View,
} from "react-native";
import * as Location from "expo-location";

import { isExpoLocationPermissionGranted } from "../../../core/location/locationPermissionState";
import { markNotificationDisclosureAccepted } from "../../../core/notifications/notificationDisclosurePersistence";
import { useAccessibilityScale } from "../../../design/responsive/useAccessibilityScale";
import { semanticDanger, semanticSuccess, semanticWarning } from "../../../design/responsive/colors";
import { AppText } from "../../../design/ui/AppText";
import { createShadow } from "../../../styles/shadowStyles";
import {
  checkBatteryOptimizationStatus,
  getOemBatteryGuidance,
  openOemBatterySettings,
  requestIgnoreBatteryOptimizations,
} from "../services/batteryOptimization";
import {
  openMissionLiveTrackingDisclosureForReadiness,
} from "../services/missionLiveTrackingDisclosureBridge";
import {
  markTrackingOnboarded,
  setTrackingNeedsAttention,
} from "../services/trackingReadinessPersistence";
import { D } from "../theme/driverDashboardTheme";

const primaryButtonShadow = createShadow({
  shadowColor: D.brandDark,
  shadowOffset: { width: 0, height: 2 },
  shadowOpacity: 0.18,
  shadowRadius: 4,
  elevation: 3,
});

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

  const fgPermissionGranted = isExpoLocationPermissionGranted(fg);
  const bgPermissionGranted = isExpoLocationPermissionGranted(bg);
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
  /** Permet de masquer le panneau pour la session (bouton de fermeture). */
  onDismiss?: () => void;
};

export function DriverTrackingReadinessGate(props: Props) {
  const { onReadyChange, silent, onDismiss } = props;
  const { shouldStackRows } = useAccessibilityScale();
  const [loading, setLoading] = useState(true);
  const [snapshot, setSnapshot] = useState<TrackingReadinessSnapshot | null>(null);
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

  const requestBgWithDisclosure = useCallback(() => {
    openMissionLiveTrackingDisclosureForReadiness(() => {
      void refresh();
    });
  }, [refresh]);

  const requestNotifications = useCallback(() => {
    void (async () => {
      try {
        await markNotificationDisclosureAccepted();
        if (Platform.OS !== "web") {
          const Notifications = await import("expo-notifications");
          await Notifications.requestPermissionsAsync();
        }
      } catch {
        /* best-effort — l'utilisateur peut passer par Ouvrir réglages */
      }
      await refresh();
    })();
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

  const primaryActions = useMemo(() => {
    const actions: Array<{ key: string; label: string; onPress: () => void }> = [
      {
        key: "location",
        label: "Localisation",
        onPress: () => void requestBgWithDisclosure(),
      },
    ];
    if (!snapshot?.notificationsGranted) {
      actions.push({
        key: "notifications",
        label: "Notifications",
        onPress: requestNotifications,
      });
    }
    actions.push({
      key: "battery",
      label: "Batterie",
      onPress: () => void requestIgnoreBatteryOptimizations(),
    });
    if (snapshot?.hasOemSettings) {
      actions.push({
        key: "oem",
        label: "Fabricant",
        onPress: () => void openOemBatterySettings(),
      });
    }
    return actions;
  }, [requestBgWithDisclosure, requestNotifications, snapshot?.hasOemSettings, snapshot?.notificationsGranted]);

  const primaryRows = useMemo(() => {
    if (shouldStackRows) {
      return primaryActions.map((action) => [action]);
    }
    const rows: Array<typeof primaryActions> = [];
    for (let i = 0; i < primaryActions.length; i += 2) {
      rows.push(primaryActions.slice(i, i + 2));
    }
    return rows;
  }, [primaryActions, shouldStackRows]);

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
      <View style={styles.headerRow}>
        <AppText variant="sectionTitle" style={styles.title} scaleRole="content">
          Préparation tracking obligatoire
        </AppText>
        {onDismiss ? (
          <Pressable
            onPress={onDismiss}
            style={styles.closeButton}
            hitSlop={10}
            accessibilityRole="button"
            accessibilityLabel="Fermer le panneau de préparation"
          >
            <AppText variant="label" style={styles.closeButtonText} scaleRole="chrome">
              ✕
            </AppText>
          </Pressable>
        ) : null}
      </View>
      <AppText variant="bodyMuted" style={styles.subtitle} scaleRole="content">
        Avant votre première mission, vérifiez les réglages ci-dessous.
      </AppText>
      {checklist.map((item) => (
        <AppText
          key={item.label}
          variant="body"
          scaleRole="content"
          style={[styles.checkItem, { color: item.ok ? semanticSuccess.fg : semanticDanger.fg }]}
        >
          {item.ok ? "✓" : "✗"} {item.label}
        </AppText>
      ))}
      {snapshot?.hasOemSettings ? (
        <AppText variant="caption" style={styles.oemHint} scaleRole="content">
          Fabricant détecté ({snapshot.oem}). Ouvrez aussi les réglages avancés du fabricant
          (auto-start / apps protégées).
        </AppText>
      ) : null}
      <View style={styles.actions}>
        {primaryRows.map((row) => (
          <View
            key={row.map((a) => a.key).join("-")}
            style={[styles.actionsRow, shouldStackRows && styles.actionsRowStacked]}
          >
            {row.map((action) => (
              <Pressable
                key={action.key}
                style={({ pressed }) => [
                  styles.buttonPrimary,
                  row.length === 1 || shouldStackRows ? styles.buttonFull : styles.buttonHalf,
                  pressed && styles.buttonPressed,
                ]}
                onPress={action.onPress}
                accessibilityRole="button"
                accessibilityLabel={action.label}
              >
                <AppText variant="label" style={styles.buttonPrimaryText} scaleRole="chrome">
                  {action.label}
                </AppText>
              </Pressable>
            ))}
          </View>
        ))}
        <View style={[styles.actionsRow, shouldStackRows && styles.actionsRowStacked]}>
          <Pressable
            style={({ pressed }) => [
              styles.buttonSecondary,
              shouldStackRows ? styles.buttonFull : styles.buttonHalf,
              pressed && styles.buttonSecondaryPressed,
            ]}
            onPress={() => Linking.openSettings()}
            accessibilityRole="button"
            accessibilityLabel="Ouvrir les réglages système"
          >
            <AppText variant="label" style={styles.buttonSecondaryText} scaleRole="chrome">
              Réglages
            </AppText>
          </Pressable>
          <Pressable
            style={({ pressed }) => [
              styles.buttonSecondary,
              shouldStackRows ? styles.buttonFull : styles.buttonHalf,
              pressed && styles.buttonSecondaryPressed,
            ]}
            onPress={() => void refresh()}
            accessibilityRole="button"
            accessibilityLabel="Revérifier les prérequis tracking"
          >
            <AppText variant="label" style={styles.buttonSecondaryText} scaleRole="chrome">
              Revérifier
            </AppText>
          </Pressable>
        </View>
      </View>
      <View style={[styles.warningBox, { backgroundColor: semanticWarning.bg }]}>
        <AppText variant="body" style={{ color: semanticWarning.fg }} scaleRole="content">
          Le démarrage d&apos;une mission suivie (écran verrouillé) requiert ces réglages.
          Vous pouvez continuer à consulter vos missions.
        </AppText>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    padding: 16,
    gap: 10,
  },
  headerRow: {
    flexDirection: "row",
    alignItems: "flex-start",
    justifyContent: "space-between",
    gap: 12,
  },
  title: {
    flex: 1,
    fontWeight: "700",
  },
  closeButton: {
    width: 28,
    height: 28,
    borderRadius: 14,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "rgba(15, 23, 42, 0.06)",
    marginTop: 2,
  },
  closeButtonText: {
    fontWeight: "700",
    color: "#475569",
  },
  subtitle: {
    opacity: 0.8,
  },
  checkItem: {
    fontWeight: "600",
  },
  oemHint: {
    opacity: 0.85,
  },
  actions: {
    gap: 10,
    marginTop: 10,
  },
  actionsRow: {
    flexDirection: "row",
    gap: 10,
    alignItems: "stretch",
  },
  actionsRowStacked: {
    flexDirection: "column",
  },
  buttonHalf: {
    flex: 1,
    minWidth: 0,
  },
  buttonFull: {
    flex: 1,
    alignSelf: "stretch",
  },
  buttonPrimary: {
    minHeight: 48,
    backgroundColor: D.brandCta,
    borderRadius: 12,
    paddingVertical: 12,
    paddingHorizontal: 12,
    alignItems: "center",
    justifyContent: "center",
    borderWidth: 1,
    borderColor: D.brandDark,
    ...primaryButtonShadow,
  },
  buttonPrimaryText: {
    color: "#FFFFFF",
    fontWeight: "700",
    textAlign: "center",
  },
  buttonPressed: {
    opacity: 0.88,
  },
  buttonSecondary: {
    minHeight: 48,
    borderRadius: 12,
    paddingVertical: 12,
    paddingHorizontal: 12,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#FFFFFF",
    borderWidth: 1.5,
    borderColor: "rgba(10, 106, 97, 0.35)",
  },
  buttonSecondaryPressed: {
    backgroundColor: "rgba(10, 106, 97, 0.08)",
    opacity: 0.95,
  },
  buttonSecondaryText: {
    color: D.brandCta,
    fontWeight: "700",
    textAlign: "center",
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
  },
  readyBody: {
    color: semanticSuccess.fg,
  },
});
