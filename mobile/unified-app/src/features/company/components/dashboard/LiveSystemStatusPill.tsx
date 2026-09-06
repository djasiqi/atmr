import { useEffect, useState } from "react";
import { FONT_SIZE } from "../../../../design/responsive/typographyTokens";

import { Platform, Pressable, StyleSheet, View } from "react-native";

import Animated, {

  Easing,

  useAnimatedStyle,

  useSharedValue,

  withRepeat,

  withTiming,

} from "react-native-reanimated";

import { AppText } from "../../../../design/ui/AppText";

import {

  cockpitDataFreshnessAccent,

  cockpitLiveStatusColor,

  cockpitLiveStatusLabel,

  type CockpitLiveStatus,

} from "../../dashboard/cockpit/cockpitLiveStatus";
import type { CompanyDataFreshness } from "../../realtime/companyRealtimeState";
import { fleetGlassPanel } from "../maps/fleetMapUiTokens";
import { E } from "../../theme/enterpriseOpsTheme";
import {
  formatGpsCoverageA11y,
  formatGpsCoverageRatio,
  type GpsCoverageCounts,
} from "./liveGpsCoverage";



const SLOW_CONNECT_THRESHOLD_MS = 7_000;
/** Aligné pilule date / chip mode (`EnterpriseHeader` HEADER_ROW_MIN_HEIGHT). */
const LIVE_PILL_HEIGHT_PX = 39;



type Props = {
  status: CockpitLiveStatus;
  /** false = pas de socket company attendu (repli HTTP). */
  realtimeSocketExpected?: boolean;
  /** Sous-titre discret (activité calme / données anciennes) — transport reste LIVE. */
  activityHint?: string | null;
  dataFreshness?: CompanyDataFreshness;
  onPress?: () => void;
  animationIntensity?: number;
  /**
   * `header` : une ligne (LIVE + couverture GPS), largeur contrainte par la rangée — hint uniquement en a11y.
   * `default` : pilule autonome (ex. overlay bas de carte).
   */
  variant?: "default" | "header";
  /** Couverture GPS N/T — même sujet que LIVE, jamais dans la barre opérationnelle. */
  gpsCoverage?: GpsCoverageCounts | null;
  /** false = roster pas encore résolu — on n’affiche pas 0/0. Une fois résolu, 0/T et T/T restent visibles. */
  gpsCoverageResolved?: boolean;
};



export function LiveSystemStatusPill({

  status,

  realtimeSocketExpected = true,

  activityHint = null,

  dataFreshness,

  onPress,
  animationIntensity = 0.35,
  variant = "default",
  gpsCoverage = null,
  gpsCoverageResolved = false,
}: Props) {
  const isHeader = variant === "header";

  const pulse = useSharedValue(0.35);

  const [isSlowConnect, setIsSlowConnect] = useState(false);

  const statusOptions = { realtimeSocketExpected };



  useEffect(() => {

    setIsSlowConnect(false);

    if (!realtimeSocketExpected || status !== "initializing") return;

    const t = setTimeout(() => setIsSlowConnect(true), SLOW_CONNECT_THRESHOLD_MS);

    return () => clearTimeout(t);

  }, [status, realtimeSocketExpected]);



  const freshnessAccent =
    dataFreshness != null ? cockpitDataFreshnessAccent(status, dataFreshness) : null;



  const color = isSlowConnect

    ? "#F59E0B"

    : cockpitLiveStatusColor(status, statusOptions);

  const label = isSlowConnect

    ? "Connexion lente…"

    : cockpitLiveStatusLabel(status, statusOptions);



  const dotColor = freshnessAccent ?? color;



  useEffect(() => {

    if (status !== "connected" || animationIntensity <= 0) {

      pulse.value = 0.3;

      return;

    }

    pulse.value = withRepeat(

      withTiming(0.85, { duration: 1400, easing: Easing.inOut(Easing.ease) }),

      -1,

      true

    );

  }, [status, animationIntensity, pulse]);



  const dotStyle = useAnimatedStyle(() => ({

    opacity: status === "connected" ? 0.45 + pulse.value * 0.45 : 1,

    transform: [{ scale: status === "connected" ? 0.92 + pulse.value * 0.12 : 1 }],

  }));



  const showCoverage = gpsCoverageResolved && gpsCoverage != null;
  const coverageA11y = showCoverage
    ? formatGpsCoverageA11y(gpsCoverage.liveCount, gpsCoverage.totalCount)
    : null;

  const a11yLabel = [activityHint ? `Statut temps réel : ${label}. ${activityHint}` : `Statut temps réel : ${label}`, coverageA11y]
    .filter(Boolean)
    .join(". ");

  const body = (
    <View
      style={[
        s.pill,
        showCoverage && !isHeader && s.pillWithCoverage,
        isHeader && s.pillHeader,
        isHeader && showCoverage && s.pillHeaderCoverage,
        fleetGlassPanel(s.glass),
      ]}
    >
      <View style={s.row}>
        <Animated.View style={[s.dot, { backgroundColor: dotColor }, dotStyle]} />
        <AppText
          variant="caption"
          style={[s.label, isHeader && s.labelHeader, { color }]}
          numberOfLines={1}
          ellipsizeMode="tail"
        >
          {label}
        </AppText>
        {showCoverage && gpsCoverage ? (
          <>
            <View style={s.separator} accessibilityElementsHidden />
            <View style={s.coverageCluster}>
              <AppText variant="caption" style={s.coverageRatio} numberOfLines={1}>
                {formatGpsCoverageRatio(gpsCoverage.liveCount, gpsCoverage.totalCount)}
              </AppText>
              {isHeader ? null : (
                <AppText variant="caption" style={s.coverageHint} numberOfLines={1}>
                  en direct
                </AppText>
              )}
            </View>
          </>
        ) : null}
        {!isHeader && !showCoverage && activityHint ? (
          <AppText variant="caption" style={s.hintInline} numberOfLines={1} ellipsizeMode="tail">
            · {activityHint}
          </AppText>
        ) : null}
      </View>
    </View>
  );



  if (!onPress) return body;



  return (

    <Pressable
      onPress={onPress}
      style={({ pressed }) => [
        isHeader && s.pressableHeader,
        pressed && s.pressed,
      ]}
      accessibilityRole="button"
      accessibilityLabel={a11yLabel}
    >

      {body}

    </Pressable>

  );

}



const s = StyleSheet.create({

  pill: {
    height: LIVE_PILL_HEIGHT_PX,
    minHeight: LIVE_PILL_HEIGHT_PX,
    maxHeight: LIVE_PILL_HEIGHT_PX,
    paddingHorizontal: 10,
    paddingVertical: 0,
    borderRadius: 16,
    justifyContent: "center",
    alignSelf: "flex-start",
    maxWidth: 168,
    overflow: "hidden",
  },
  pillWithCoverage: {
    maxWidth: 240,
    paddingHorizontal: 12,
  },
  pillHeader: {
    width: "100%",
    maxWidth: "100%",
    alignSelf: "stretch",
    flexShrink: 1,
    minWidth: 0,
  },
  pillHeaderCoverage: {
    paddingHorizontal: 12,
  },
  pressableHeader: {
    width: "100%",
    maxWidth: "100%",
    minWidth: 0,
    flexShrink: 1,
  },

  row: {
    flexDirection: "row",
    flexWrap: "nowrap",
    alignItems: "center",
    gap: 6,
    minWidth: 0,
    flexShrink: 1,
  },

  glass: {
    borderRadius: 16,

    ...Platform.select({

      web: { backdropFilter: "blur(12px)" } as object,

      default: {},

    }),

  },

  dot: {

    width: 8,

    height: 8,

    borderRadius: 4,
    flexShrink: 0,

  },

  label: {
    flexShrink: 1,
    minWidth: 0,
    fontWeight: "800",
    fontSize: FONT_SIZE.px11,
    letterSpacing: 0.6,
    textTransform: "uppercase",
  },
  labelHeader: {
    flex: 1,
    flexShrink: 1,
    minWidth: 0,
  },
  separator: {
    width: StyleSheet.hairlineWidth,
    alignSelf: "stretch",
    minHeight: 12,
    marginHorizontal: 2,
    flexShrink: 0,
    backgroundColor: "rgba(148, 163, 184, 0.42)",
  },
  coverageCluster: {
    flexDirection: "row",
    alignItems: "baseline",
    gap: 4,
    flexGrow: 0,
    flexShrink: 0,
  },
  coverageRatio: {
    flexShrink: 0,
    fontSize: FONT_SIZE.px12,
    fontWeight: "700",
    color: E.TEXT,
    letterSpacing: 0.15,
    textTransform: "none",
  },
  coverageHint: {
    flexShrink: 1,
    minWidth: 0,
    fontSize: FONT_SIZE.px10,
    fontWeight: "600",
    color: E.TEXT_SEC,
    letterSpacing: 0.1,
    textTransform: "none",
  },

  hintInline: {
    flex: 1,
    flexShrink: 1,
    minWidth: 0,
    fontSize: FONT_SIZE.px10,
    fontWeight: "600",
    color: "#64748B",
    letterSpacing: 0.15,
    textTransform: "none",
  },

  pressed: { opacity: 0.9 },

});


