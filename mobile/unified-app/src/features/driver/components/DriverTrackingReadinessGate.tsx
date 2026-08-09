/**
 * Panneau pédagogique tracking — informatif, non bloquant pour la navigation.
 *
 * Assistant de correction contextuel : n'affiche que les actions encore nécessaires.
 * Ne confond pas non vérifiable / non configuré / non applicable.
 *
 * NB : la lecture des prérequis device (permissions Location, GPS) est
 * faite directement via expo-location, indépendamment du feature flag
 * `tracking_background_enabled`.
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

import {
  isExpoLocationPermissionGranted,
  resolveLocationAccuracy,
} from "../../../core/location/locationPermissionState";
import { markNotificationDisclosureAccepted } from "../../../core/notifications/notificationDisclosurePersistence";
import { requestNotificationOsPermissionsAsync } from "../../../core/notifications/requestNotificationOsPermissions";
import { emitDriverTelemetry } from "../../../core/observability/driverTelemetry";
import { useAccessibilityScale } from "../../../design/responsive/useAccessibilityScale";
import {
  semanticDanger,
  semanticSuccess,
  semanticWarning,
} from "../../../design/responsive/colors";
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
  type ReadinessLocationAction,
} from "../services/missionLiveTrackingDisclosureBridge";
import {
  isOemGuidanceAcknowledgedFor,
  markOemGuidanceAcknowledged,
} from "../services/oemGuidancePersistence";
import {
  markTrackingOnboarded,
  setTrackingNeedsAttention,
} from "../services/trackingReadinessPersistence";
import {
  batteryActionLabel,
  computeTrackingReady,
  locationActionLabel,
  resolveBatteryReadinessStatus,
  resolveLocationReadinessAction,
  shouldApplyRefreshSequence,
  shouldShowOemGuidance,
  type TrackingReadinessSnapshot,
} from "../services/trackingReadinessModel";
import { D } from "../theme/driverDashboardTheme";

export type { TrackingReadinessSnapshot };

const primaryButtonShadow = createShadow({
  shadowColor: D.brandDark,
  shadowOffset: { width: 0, height: 2 },
  shadowOpacity: 0.18,
  shadowRadius: 4,
  elevation: 3,
});

async function readNotificationsGranted(): Promise<boolean> {
  if (Platform.OS === "web") return true;
  try {
    // require : compatible mocks Jest (import() dynamique peut contourner le mock).
    // eslint-disable-next-line @typescript-eslint/no-require-imports -- module optionnel runtime
    const Notifications = require("expo-notifications") as {
      getPermissionsAsync: () => Promise<{ granted?: boolean; status?: string }>;
    };
    const perm = await Notifications.getPermissionsAsync();
    return Boolean(perm.granted || perm.status === "granted");
  } catch {
    return false;
  }
}

async function openLocationServicesSettings(): Promise<void> {
  if (Platform.OS === "android") {
    try {
      // eslint-disable-next-line @typescript-eslint/no-require-imports -- Android-only
      const IntentLauncher = require("expo-intent-launcher") as {
        startActivityAsync: (action: string, params?: Record<string, unknown>) => Promise<unknown>;
        ActivityAction?: { LOCATION_SOURCE_SETTINGS?: string };
      };
      const action =
        IntentLauncher.ActivityAction?.LOCATION_SOURCE_SETTINGS ??
        "android.settings.LOCATION_SOURCE_SETTINGS";
      await IntentLauncher.startActivityAsync(action);
      return;
    } catch {
      /* fallback réglages app */
    }
  }
  if (Platform.OS === "ios") {
    await Linking.openURL("app-settings:").catch(() => Linking.openSettings());
    return;
  }
  await Linking.openSettings();
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
  const locationAccuracy = resolveLocationAccuracy(fg);
  const gpsEnabled = Boolean(servicesEnabled);
  const batteryStatus = resolveBatteryReadinessStatus({
    platformOs: Platform.OS,
    checked: battery.checked,
    isIgnoring: battery.isIgnoring,
  });
  const batteryExempt =
    batteryStatus === "exempt" || batteryStatus === "not_applicable";
  const oemGuidanceAcknowledged = await isOemGuidanceAcknowledgedFor(oem.oem);

  const ready = computeTrackingReady({
    fgPermissionGranted,
    bgPermissionGranted,
    locationAccuracy,
    gpsEnabled,
    notificationsGranted,
    batteryStatus,
  });

  return {
    ready,
    bgPermissionGranted,
    fgPermissionGranted,
    notificationsGranted,
    batteryExempt,
    batteryStatus,
    locationAccuracy,
    gpsEnabled,
    oem: oem.oem,
    hasOemSettings: oem.hasOemSettings,
    oemGuidanceAcknowledged,
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

type GateAction = {
  key: string;
  label: string;
  onPress: () => void;
  variant: "primary" | "secondary";
};

type ChecklistTone = "ok" | "bad" | "warn" | "na";

type ChecklistItem = {
  key: string;
  label: string;
  tone: ChecklistTone;
};

function chunkActions<T>(items: T[], size: number): T[][] {
  const rows: T[][] = [];
  for (let i = 0; i < items.length; i += size) {
    rows.push(items.slice(i, i + size));
  }
  return rows;
}

function toneColor(tone: ChecklistTone): string {
  switch (tone) {
    case "ok":
      return semanticSuccess.fg;
    case "bad":
      return semanticDanger.fg;
    case "warn":
      return semanticWarning.fg;
    case "na":
      return "#64748B";
  }
}

function toneMark(tone: ChecklistTone): string {
  if (tone === "ok") return "✓";
  if (tone === "na") return "–";
  if (tone === "warn") return "!";
  return "✗";
}

export function DriverTrackingReadinessGate(props: Props) {
  const { onReadyChange, silent, onDismiss } = props;
  const { fontScale, isVeryLargeText } = useAccessibilityScale();
  /** Empilement uniquement si police vraiment extrême ; sinon grille 2–3 colonnes (chrome capped). */
  const stackActions = fontScale >= 1.75;
  const columnsPerRow = stackActions ? 1 : isVeryLargeText ? 2 : 3;
  const [loading, setLoading] = useState(true);
  const [snapshot, setSnapshot] = useState<TrackingReadinessSnapshot | null>(null);
  const onboardedRef = useRef(false);
  const refreshSequenceRef = useRef(0);

  const refresh = useCallback(async () => {
    const sequence = ++refreshSequenceRef.current;
    setLoading(true);
    try {
      const next = await evaluateTrackingReadiness();
      if (!shouldApplyRefreshSequence(sequence, refreshSequenceRef.current)) {
        return;
      }
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
    } finally {
      if (shouldApplyRefreshSequence(sequence, refreshSequenceRef.current)) {
        setLoading(false);
      }
    }
  }, [onReadyChange]);

  useEffect(() => {
    void refresh();
    const sub = AppState.addEventListener("change", (next) => {
      if (next === "active") void refresh();
    });
    return () => sub.remove();
  }, [refresh]);

  const runLocationAction = useCallback(
    (action: Exclude<ReturnType<typeof resolveLocationReadinessAction>, null>) => {
      if (action === "enable_precise" || action === "verify_accuracy") {
        emitDriverTelemetry("tracking.readiness.action.location_accuracy", {
          source: "driver.tracking_readiness_gate",
          reason: action,
        });
        void (async () => {
          if (action === "enable_precise") {
            try {
              const fg = await Location.requestForegroundPermissionsAsync();
              if (resolveLocationAccuracy(fg) === "precise") {
                await refresh();
                return;
              }
            } catch {
              /* ouvrir les réglages ci-dessous */
            }
          }
          await Linking.openSettings().catch(() => undefined);
          await refresh();
        })();
        return;
      }

      const bridgeAction: ReadinessLocationAction =
        action === "background" ? "background" : "foreground";

      emitDriverTelemetry(
        bridgeAction === "foreground"
          ? "tracking.readiness.action.location_foreground"
          : "tracking.readiness.action.location_background",
        { source: "driver.tracking_readiness_gate" }
      );

      openMissionLiveTrackingDisclosureForReadiness(() => {
        void refresh();
      }, bridgeAction);
    },
    [refresh]
  );

  const requestNotifications = useCallback(() => {
    emitDriverTelemetry("tracking.readiness.action.notifications", {
      source: "driver.tracking_readiness_gate",
    });
    void (async () => {
      try {
        await markNotificationDisclosureAccepted();
        if (Platform.OS !== "web") {
          await requestNotificationOsPermissionsAsync();
        }
      } catch {
        /* best-effort — l'utilisateur peut passer par Ouvrir réglages */
      }
      await refresh();
    })();
  }, [refresh]);

  const requestBattery = useCallback(() => {
    emitDriverTelemetry("tracking.readiness.action.battery", {
      source: "driver.tracking_readiness_gate",
      reason: snapshot?.batteryStatus ?? null,
    });
    void requestIgnoreBatteryOptimizations().finally(() => {
      void refresh();
    });
  }, [refresh, snapshot?.batteryStatus]);

  const requestOem = useCallback(() => {
    emitDriverTelemetry("tracking.readiness.action.oem", {
      source: "driver.tracking_readiness_gate",
      oem: snapshot?.oem ?? null,
    });
    void (async () => {
      const result = await openOemBatterySettings();
      if (result.opened && snapshot?.oem) {
        await markOemGuidanceAcknowledged(snapshot.oem).catch(() => undefined);
      } else if (!result.opened) {
        await Linking.openSettings().catch(() => undefined);
      }
      await refresh();
    })();
  }, [refresh, snapshot?.oem]);

  const requestGps = useCallback(() => {
    emitDriverTelemetry("tracking.readiness.action.gps_settings", {
      source: "driver.tracking_readiness_gate",
    });
    void openLocationServicesSettings().finally(() => {
      void refresh();
    });
  }, [refresh]);

  const checklist = useMemo((): ChecklistItem[] => {
    if (!snapshot) return [];
    const accuracyTone: ChecklistTone =
      snapshot.locationAccuracy === "precise"
        ? "ok"
        : snapshot.locationAccuracy === "approximate"
          ? "bad"
          : snapshot.fgPermissionGranted
            ? "warn"
            : "bad";
    const batteryTone: ChecklistTone =
      snapshot.batteryStatus === "not_applicable"
        ? "na"
        : snapshot.batteryStatus === "exempt"
          ? "ok"
          : snapshot.batteryStatus === "restricted"
            ? "bad"
            : "warn";

    const items: ChecklistItem[] = [
      {
        key: "fg",
        label: "Localisation autorisée",
        tone: snapshot.fgPermissionGranted ? "ok" : "bad",
      },
      {
        key: "accuracy",
        label:
          snapshot.locationAccuracy === "approximate"
            ? "Position précise requise"
            : "Précision de localisation",
        tone: accuracyTone,
      },
      {
        key: "bg",
        label: "Localisation arrière-plan (Toujours)",
        tone: snapshot.bgPermissionGranted ? "ok" : "bad",
      },
      {
        key: "gps",
        label: "GPS activé",
        tone: snapshot.gpsEnabled ? "ok" : "bad",
      },
      {
        key: "notifications",
        label: "Notifications autorisées",
        tone: snapshot.notificationsGranted ? "ok" : "bad",
      },
    ];

    if (snapshot.batteryStatus !== "not_applicable") {
      items.push({
        key: "battery",
        label:
          snapshot.batteryStatus === "unknown"
            ? "Batterie à vérifier"
            : "Optimisation batterie désactivée",
        tone: batteryTone,
      });
    }

    return items;
  }, [snapshot]);

  const showOem = snapshot
    ? shouldShowOemGuidance({
        hasOemSettings: snapshot.hasOemSettings,
        oemGuidanceAcknowledged: snapshot.oemGuidanceAcknowledged,
        batteryStatus: snapshot.batteryStatus,
      })
    : false;

  const gateActions = useMemo((): GateAction[] => {
    const actions: GateAction[] = [];
    if (!snapshot) return actions;

    const locationAction = resolveLocationReadinessAction({
      fgPermissionGranted: snapshot.fgPermissionGranted,
      bgPermissionGranted: snapshot.bgPermissionGranted,
      locationAccuracy: snapshot.locationAccuracy,
    });
    if (locationAction) {
      actions.push({
        key: `location_${locationAction}`,
        label: locationActionLabel(locationAction),
        onPress: () => runLocationAction(locationAction),
        variant: "primary",
      });
    }

    if (!snapshot.gpsEnabled) {
      actions.push({
        key: "gps",
        label: "Activer le GPS",
        onPress: requestGps,
        variant: "primary",
      });
    }

    if (!snapshot.notificationsGranted) {
      actions.push({
        key: "notifications",
        label: "Notifications",
        onPress: requestNotifications,
        variant: "primary",
      });
    }

    const batteryLabel = batteryActionLabel(snapshot.batteryStatus);
    if (batteryLabel) {
      actions.push({
        key: "battery",
        label: batteryLabel,
        onPress: requestBattery,
        variant: "primary",
      });
    }

    if (showOem) {
      actions.push({
        key: "oem",
        label: "Guide fabricant",
        onPress: requestOem,
        variant: "primary",
      });
    }

    actions.push({
      key: "settings",
      label: "Réglages",
      onPress: () => Linking.openSettings(),
      variant: "secondary",
    });

    return actions;
  }, [
    snapshot,
    showOem,
    runLocationAction,
    requestGps,
    requestNotifications,
    requestBattery,
    requestOem,
  ]);

  const actionRows = useMemo(
    () => chunkActions(gateActions, columnsPerRow),
    [columnsPerRow, gateActions]
  );

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
        Corrigez uniquement les éléments manquants ci-dessous.
      </AppText>
      {checklist.map((item) => (
        <AppText
          key={item.key}
          variant="body"
          scaleRole="content"
          style={[styles.checkItem, { color: toneColor(item.tone) }]}
        >
          {toneMark(item.tone)} {item.label}
        </AppText>
      ))}
      {showOem ? (
        <AppText variant="caption" style={styles.oemHint} scaleRole="content">
          Guide fabricant ({snapshot?.oem}). Ouvrez les réglages avancés (auto-start /
          apps protégées) — l’application ne peut pas confirmer ce réglage
          techniquement.
        </AppText>
      ) : null}
      <View style={styles.actions}>
        {actionRows.map((row) => (
          <View
            key={row.map((a) => a.key).join("-")}
            style={[styles.actionsRow, stackActions && styles.actionsRowStacked]}
          >
            {row.map((action) => {
              const isPrimary = action.variant === "primary";
              return (
                <Pressable
                  key={action.key}
                  style={({ pressed }) => [
                    isPrimary ? styles.buttonPrimary : styles.buttonSecondary,
                    stackActions || row.length === 1 ? styles.buttonFull : styles.buttonCell,
                    pressed && (isPrimary ? styles.buttonPressed : styles.buttonSecondaryPressed),
                  ]}
                  onPress={action.onPress}
                  accessibilityRole="button"
                  accessibilityLabel={action.label}
                >
                  <AppText
                    variant="label"
                    style={isPrimary ? styles.buttonPrimaryText : styles.buttonSecondaryText}
                    scaleRole="chrome"
                    numberOfLines={2}
                  >
                    {action.label}
                  </AppText>
                </Pressable>
              );
            })}
          </View>
        ))}
      </View>
      <Pressable
        onPress={() => void refresh()}
        hitSlop={8}
        accessibilityRole="button"
        accessibilityLabel="Revérifier l’état du tracking"
        style={styles.recheckLink}
      >
        <AppText variant="caption" style={styles.recheckLinkText} scaleRole="chrome">
          État incorrect ? Revérifier
        </AppText>
      </Pressable>
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
    gap: 6,
    marginTop: 8,
  },
  actionsRow: {
    flexDirection: "row",
    gap: 6,
    alignItems: "stretch",
  },
  actionsRowStacked: {
    flexDirection: "column",
  },
  buttonCell: {
    flex: 1,
    minWidth: 0,
  },
  buttonFull: {
    flex: 1,
    alignSelf: "stretch",
  },
  buttonPrimary: {
    minHeight: 36,
    backgroundColor: D.brandCta,
    borderRadius: 10,
    paddingVertical: 8,
    paddingHorizontal: 8,
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
    fontSize: 12,
  },
  buttonPressed: {
    opacity: 0.88,
  },
  buttonSecondary: {
    minHeight: 36,
    borderRadius: 10,
    paddingVertical: 8,
    paddingHorizontal: 8,
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
    fontSize: 12,
  },
  recheckLink: {
    alignSelf: "center",
    paddingVertical: 4,
    marginTop: 2,
  },
  recheckLinkText: {
    color: "#64748B",
    textDecorationLine: "underline",
  },
});
