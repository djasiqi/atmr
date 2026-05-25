import { memo, useEffect, useMemo, useRef, useState } from "react";
import {
  Animated,
  Easing,
  StyleSheet,
  View,
  type ViewStyle,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import LottieView from "lottie-react-native";
import * as Location from "expo-location";
import { AppText } from "../../../design/ui/AppText";
import { createShadow } from "../../../styles/shadowStyles";
import { D, dashboardCardShadow } from "../theme/driverDashboardTheme";
import { FONT_SIZE } from "../../../design/responsive/typographyTokens";
import { useSocketStatus } from "../hooks/useSocketStatus";
import { useTrackingState } from "../hooks/useTrackingState";
import type { DriverMission } from "../types";

const driverIdleCarLottie = require("../../../../assets/lottie/driver/driver-idle-car.json");

const idleCardShadow = createShadow(dashboardCardShadow);

const TIP_BG = "rgba(0, 121, 107, 0.07)";

const IDLE_TIPS = [
  "Votre position est bien partagée avec le dispatch.",
  "Les nouvelles missions apparaîtront automatiquement.",
  "Gardez l'application ouverte pour une synchronisation optimale.",
];

const TIP_ROTATE_INTERVAL_MS = 30_000;
type Props = {
  /** Disponibilité chauffeur (driver `is_available`). */
  isAvailable: boolean;
  /** Bookings du jour utilisés pour dériver les stats — facultatif. */
  todayMissions?: DriverMission[];
};

export const DriverIdleState = memo(function DriverIdleState({
  isAvailable,
  todayMissions,
}: Props) {
  const tracking = useTrackingState();
  const socket = useSocketStatus();
  const tipIndex = useTipRotation();

  const [gpsEnabled, setGpsEnabled] = useState(true);
  useEffect(() => {
    let mounted = true;
    void Location.hasServicesEnabledAsync()
      .then((enabled) => {
        if (mounted) setGpsEnabled(enabled);
      })
      .catch(() => {
        if (mounted) setGpsEnabled(false);
      });
    return () => {
      mounted = false;
    };
  }, [tracking.isTracking, tracking.lastUpdate]);

  const gpsConnected = gpsEnabled && tracking.isTracking;
  const realtimeConnected = socket.connected && !socket.degraded;
  const todayStats = useMemo(
    () => deriveTodayStats(todayMissions ?? []),
    [todayMissions]
  );

  return (
    <View style={styles.root} accessibilityLabel="État chauffeur disponible">
      <View style={styles.mainCard}>
        <RadarPulse active={isAvailable && (realtimeConnected || gpsConnected)} />
        <View style={styles.headerTextBlock}>
          <AppText variant="sectionTitle" style={styles.title}>
            Prêt à recevoir une mission
          </AppText>
          <AppText variant="bodyMuted" style={styles.subtitle}>
            Vous êtes disponible et connecté. Les nouvelles missions apparaîtront
            automatiquement.
          </AppText>
        </View>

        <View style={styles.summaryDivider} accessibilityElementsHidden />

        <AppText variant="caption" style={styles.summaryHeaderLabel}>
          Aujourd’hui
        </AppText>

        <View style={styles.summaryMetricsRow}>
          <TodayMetric
            value={`${todayStats.completedMissions}`}
            label={
              todayStats.completedMissions === 1
                ? "Mission réalisée"
                : "Missions réalisées"
            }
          />
          <TodayMetric
            value={formatDistanceKm(todayStats.distanceKm)}
            label="Distance parcourue"
            divider
          />
          <TodayMetric
            value={formatDrivingTime(todayStats.drivingTimeMinutes)}
            label="Temps de travail"
            divider
          />
        </View>
      </View>

      <View style={styles.tipCard} accessibilityRole="text">
        <View style={styles.tipIconWrap} accessibilityElementsHidden>
          <Ionicons name="bulb-outline" size={14} color={D.brand} />
        </View>
        <View style={styles.tipTextCol}>
          <AppText variant="caption" style={styles.tipTitle}>
            Conseil du jour
          </AppText>
          <AppText variant="caption" style={styles.tipBody} numberOfLines={2}>
            {IDLE_TIPS[tipIndex]}
          </AppText>
        </View>
      </View>
    </View>
  );
});

function TodayMetric(props: {
  value: string;
  label: string;
  divider?: boolean;
}) {
  return (
    <>
      {props.divider ? (
        <View style={styles.todayMetricDivider} accessibilityElementsHidden />
      ) : null}
      <View style={styles.todayMetric}>
        <AppText variant="sectionTitle" style={styles.todayMetricValue} numberOfLines={1}>
          {props.value}
        </AppText>
        <AppText
          variant="caption"
          style={styles.todayMetricLabel}
          numberOfLines={2}
        >
          {props.label}
        </AppText>
      </View>
    </>
  );
}

const RADAR_PULSE_DURATION_MS = 2800;
const RADAR_RING_COUNT = 3;
const BREATH_DURATION_MS = 2400;

function RadarPulse({ active }: { active: boolean }) {
  const ringValuesRef = useRef<Animated.Value[]>(
    Array.from({ length: RADAR_RING_COUNT }, () => new Animated.Value(0))
  );
  const breath = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    const ringValues = ringValuesRef.current;
    if (!active) {
      ringValues.forEach((v) => v.setValue(0));
      breath.setValue(0);
      return;
    }
    const ringLoops = ringValues.map((val, idx) =>
      Animated.loop(
        Animated.sequence([
          Animated.delay((RADAR_PULSE_DURATION_MS / RADAR_RING_COUNT) * idx),
          Animated.timing(val, {
            toValue: 1,
            duration: RADAR_PULSE_DURATION_MS,
            easing: Easing.out(Easing.cubic),
            useNativeDriver: true,
          }),
          Animated.timing(val, {
            toValue: 0,
            duration: 0,
            useNativeDriver: true,
          }),
        ])
      )
    );

    const breathLoop = Animated.loop(
      Animated.sequence([
        Animated.timing(breath, {
          toValue: 1,
          duration: BREATH_DURATION_MS,
          easing: Easing.inOut(Easing.sin),
          useNativeDriver: true,
        }),
        Animated.timing(breath, {
          toValue: 0,
          duration: BREATH_DURATION_MS,
          easing: Easing.inOut(Easing.sin),
          useNativeDriver: true,
        }),
      ])
    );

    ringLoops.forEach((loop) => loop.start());
    breathLoop.start();
    return () => {
      ringLoops.forEach((loop) => loop.stop());
      breathLoop.stop();
    };
  }, [active, breath]);

  const ringStartScale = CAR_BUBBLE_SIZE / RADAR_SIZE;
  const ringStyle = (val: Animated.Value): ViewStyle => ({
    transform: [
      {
        scale: val.interpolate({
          inputRange: [0, 1],
          outputRange: [ringStartScale, 1],
        }),
      },
    ],
    opacity: val.interpolate({
      inputRange: [0, 0.06, 0.88, 1],
      outputRange: [0, 0.65, 0.04, 0],
    }),
  });

  const haloOuterStyle: ViewStyle = {
    transform: [
      {
        scale: breath.interpolate({
          inputRange: [0, 1],
          outputRange: [1, 1.06],
        }),
      },
    ],
    opacity: breath.interpolate({
      inputRange: [0, 1],
      outputRange: [0.7, 1],
    }),
  };

  const haloInnerStyle: ViewStyle = {
    transform: [
      {
        scale: breath.interpolate({
          inputRange: [0, 1],
          outputRange: [1.02, 0.98],
        }),
      },
    ],
    opacity: breath.interpolate({
      inputRange: [0, 1],
      outputRange: [1, 0.75],
    }),
  };

  return (
    <View style={styles.radarWrap} pointerEvents="none">
      {ringValuesRef.current.map((val, idx) => (
        <Animated.View key={idx} style={[styles.ring, ringStyle(val)]} />
      ))}
      <Animated.View style={[styles.carHaloOuter, haloOuterStyle]} />
      <Animated.View style={[styles.carHaloInner, haloInnerStyle]} />
      <View style={styles.carBubble}>
        <View style={styles.carBubbleInnerRing} accessibilityElementsHidden />
        <View accessibilityElementsHidden style={styles.carLottieWrap}>
          <LottieView
            source={driverIdleCarLottie}
            autoPlay={active}
            loop
            speed={active ? 0.55 : 0}
            style={styles.carLottie}
          />
        </View>
      </View>
    </View>
  );
}

function useTipRotation(): number {
  const [index, setIndex] = useState(0);
  useEffect(() => {
    const interval = setInterval(() => {
      setIndex((current) => (current + 1) % IDLE_TIPS.length);
    }, TIP_ROTATE_INTERVAL_MS);
    return () => clearInterval(interval);
  }, []);
  return index;
}

function deriveTodayStats(missions: DriverMission[]): {
  completedMissions: number;
  distanceKm: number;
  drivingTimeMinutes: number;
} {
  const todayKey = formatLocalDayKey(new Date());
  let completedMissions = 0;
  let distanceMeters = 0;
  let drivingMinutes = 0;

  for (const mission of missions) {
    const status = String(mission.status ?? "").toUpperCase();
    if (status !== "COMPLETED") continue;
    const completedAt = parseMissionDate(
      mission.completed_at,
      mission.ended_at,
      mission.finished_at
    );
    if (!completedAt || formatLocalDayKey(completedAt) !== todayKey) continue;

    completedMissions += 1;

    const distanceEstimated = readMissionBoolean(
      mission.distance_duration_estimated,
      mission.distance_estimated
    );
    if (!distanceEstimated) {
      const distance = pickFiniteNumber(
        mission.actual_distance_meters,
        mission.distance_meters,
        mission.distanceMeters,
        typeof mission.distance_km === "number"
          ? (mission.distance_km as number) * 1000
          : null
      );
      if (distance != null && distance > 0) {
        distanceMeters += distance;
      }
    }

    const boardedAt = parseMissionDate(
      mission.boarded_at,
      mission.in_progress_at,
      mission.onboarded_at,
      mission.started_at
    );
    const durationFromTimeline =
      boardedAt && completedAt && completedAt.getTime() > boardedAt.getTime()
        ? (completedAt.getTime() - boardedAt.getTime()) / 60_000
        : null;
    const duration = pickFiniteNumber(
      durationFromTimeline,
      mission.duree_minutes,
      mission.duration_in_minutes,
      mission.duration_minutes,
      typeof mission.duration_seconds === "number"
        ? (mission.duration_seconds as number) / 60
        : null
    );
    if (duration != null && duration > 0) {
      drivingMinutes += duration;
    }
  }

  return {
    completedMissions,
    distanceKm: distanceMeters > 0 ? distanceMeters / 1000 : 0,
    drivingTimeMinutes: Math.round(drivingMinutes),
  };
}

function parseMissionDate(...values: unknown[]): Date | null {
  for (const raw of values) {
    if (typeof raw !== "string" || !raw.trim()) continue;
    const parsed = new Date(raw);
    if (Number.isFinite(parsed.getTime())) return parsed;
  }
  return null;
}

function readMissionBoolean(...values: unknown[]): boolean {
  for (const raw of values) {
    if (typeof raw === "boolean") return raw;
    if (typeof raw === "number") return raw > 0;
    if (typeof raw === "string") {
      const normalized = raw.trim().toLowerCase();
      if (["1", "true", "yes", "on"].includes(normalized)) return true;
      if (["0", "false", "no", "off"].includes(normalized)) return false;
    }
  }
  return false;
}

function pickFiniteNumber(...values: unknown[]): number | null {
  for (const raw of values) {
    if (raw == null) continue;
    const num = typeof raw === "number" ? raw : Number(raw);
    if (Number.isFinite(num)) return num;
  }
  return null;
}

function formatLocalDayKey(date: Date): string {
  if (!Number.isFinite(date.getTime())) return "";
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-${String(date.getDate()).padStart(2, "0")}`;
}

function formatDistanceKm(km: number): string {
  if (!Number.isFinite(km) || km <= 0) return "0 km";
  if (km >= 100) return `${Math.round(km)} km`;
  if (km >= 10) return `${km.toFixed(0)} km`;
  return `${km.toFixed(1).replace(".", ",")} km`;
}

function formatDrivingTime(minutes: number): string {
  if (!Number.isFinite(minutes) || minutes <= 0) return "0 h 00";
  const totalMinutes = Math.max(0, Math.floor(minutes));
  const hours = Math.floor(totalMinutes / 60);
  const remaining = totalMinutes % 60;
  return `${hours} h ${String(remaining).padStart(2, "0")}`;
}

const RADAR_SIZE = 152;
const CAR_BUBBLE_SIZE = 72;
const CAR_HALO_SIZE = 116;
const CAR_HALO_INNER_SIZE = 90;
const CAR_BUBBLE_INNER_INSET = 5;
const RING_COLOR = "rgba(0, 121, 107, 0.42)";
const HALO_OUTER_COLOR = "rgba(0, 121, 107, 0.05)";
const HALO_INNER_COLOR = "rgba(0, 121, 107, 0.085)";
const BUBBLE_BORDER = "rgba(0, 121, 107, 0.16)";
const BUBBLE_INNER_RING = "rgba(0, 121, 107, 0.10)";

const styles = StyleSheet.create({
  root: {
    alignSelf: "stretch",
    gap: 12,
  },
  mainCard: {
    backgroundColor: D.cardBg,
    borderRadius: 24,
    paddingHorizontal: 20,
    paddingTop: 14,
    paddingBottom: 18,
    gap: 14,
    alignItems: "center",
    ...idleCardShadow,
  },
  radarWrap: {
    width: RADAR_SIZE,
    height: RADAR_SIZE,
    alignItems: "center",
    justifyContent: "center",
    marginTop: 2,
  },
  ring: {
    position: "absolute",
    width: RADAR_SIZE,
    height: RADAR_SIZE,
    borderRadius: RADAR_SIZE / 2,
    borderWidth: 1.25,
    borderColor: RING_COLOR,
  },
  carHaloOuter: {
    position: "absolute",
    width: CAR_HALO_SIZE,
    height: CAR_HALO_SIZE,
    borderRadius: CAR_HALO_SIZE / 2,
    backgroundColor: HALO_OUTER_COLOR,
  },
  carHaloInner: {
    position: "absolute",
    width: CAR_HALO_INNER_SIZE,
    height: CAR_HALO_INNER_SIZE,
    borderRadius: CAR_HALO_INNER_SIZE / 2,
    backgroundColor: HALO_INNER_COLOR,
  },
  carBubble: {
    width: CAR_BUBBLE_SIZE,
    height: CAR_BUBBLE_SIZE,
    borderRadius: CAR_BUBBLE_SIZE / 2,
    backgroundColor: D.cardBg,
    alignItems: "center",
    justifyContent: "center",
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: BUBBLE_BORDER,
    shadowColor: "#0F172A",
    shadowOpacity: 0.06,
    shadowOffset: { width: 0, height: 4 },
    shadowRadius: 10,
    elevation: 3,
  },
  carBubbleInnerRing: {
    position: "absolute",
    width: CAR_BUBBLE_SIZE - CAR_BUBBLE_INNER_INSET * 2,
    height: CAR_BUBBLE_SIZE - CAR_BUBBLE_INNER_INSET * 2,
    borderRadius: (CAR_BUBBLE_SIZE - CAR_BUBBLE_INNER_INSET * 2) / 2,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: BUBBLE_INNER_RING,
  },
  carLottieWrap: {
    width: CAR_BUBBLE_SIZE - 14,
    height: CAR_BUBBLE_SIZE - 14,
    alignItems: "center",
    justifyContent: "center",
  },
  carLottie: {
    width: "100%",
    height: "100%",
  },
  headerTextBlock: {
    alignItems: "center",
    gap: 6,
    paddingHorizontal: 4,
  },
  title: {
    color: D.text,
    fontSize: FONT_SIZE.px18,
    fontWeight: "700",
    textAlign: "center",
    lineHeight: 24,
    letterSpacing: -0.2,
  },
  subtitle: {
    color: D.textMuted,
    fontSize: FONT_SIZE.px14,
    fontWeight: "500",
    lineHeight: 20,
    textAlign: "center",
    maxWidth: 320,
  },
  summaryDivider: {
    alignSelf: "stretch",
    height: StyleSheet.hairlineWidth,
    backgroundColor: D.metricDivider,
    marginTop: 4,
  },
  summaryHeaderLabel: {
    alignSelf: "stretch",
    color: D.textMuted,
    fontSize: FONT_SIZE.px10,
    fontWeight: "700",
    letterSpacing: 1.4,
    textTransform: "uppercase",
  },
  summaryMetricsRow: {
    alignSelf: "stretch",
    flexDirection: "row",
    alignItems: "stretch",
    paddingTop: 2,
  },
  todayMetric: {
    flex: 1,
    minWidth: 0,
    alignItems: "center",
    gap: 3,
    paddingHorizontal: 6,
  },
  todayMetricValue: {
    color: D.text,
    fontSize: FONT_SIZE.px20,
    fontWeight: "700",
    lineHeight: 24,
    textAlign: "center",
    letterSpacing: -0.3,
  },
  todayMetricLabel: {
    color: D.textMuted,
    fontSize: FONT_SIZE.px11,
    fontWeight: "500",
    textAlign: "center",
    lineHeight: 14,
  },
  todayMetricDivider: {
    width: StyleSheet.hairlineWidth,
    backgroundColor: D.metricDivider,
    alignSelf: "stretch",
    marginVertical: 4,
  },
  tipCard: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
    backgroundColor: TIP_BG,
    borderRadius: 14,
    paddingHorizontal: 14,
    paddingVertical: 10,
  },
  tipIconWrap: {
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: D.cardBg,
    alignItems: "center",
    justifyContent: "center",
  },
  tipTextCol: {
    flex: 1,
    minWidth: 0,
    gap: 2,
  },
  tipTitle: {
    color: D.brand,
    fontSize: FONT_SIZE.px11,
    fontWeight: "800",
    letterSpacing: 0.4,
    lineHeight: 14,
  },
  tipBody: {
    color: D.text,
    fontSize: FONT_SIZE.px12,
    fontWeight: "500",
    lineHeight: 16,
  },
});
