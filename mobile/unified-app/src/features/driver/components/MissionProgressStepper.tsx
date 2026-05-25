import { useEffect, useRef } from "react";
import { Animated, Easing, StyleSheet, View, type DimensionValue } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { AppText } from "../../../design/ui/AppText";
import type { DriverEtaSnapshot } from "../api";
import {
  isMissionStepperSegmentComplete,
  resolveMissionStepperProgress,
} from "../domain/missionStepperProgress";
import { useMissionStepperApproachProgress } from "../hooks/useMissionStepperApproachProgress";
import type { DriverMission } from "../types";
import { D } from "../theme/driverDashboardTheme";
import { FONT_SIZE } from "../../../design/responsive/typographyTokens";

const STEPS = [
  "Assignée",
  "Arrivé patient",
  "Départ patient",
  "Terminée",
] as const;

const STEP_COUNT = STEPS.length;
const NODE_SIZE = 16;
const TRACK_HEIGHT = 18;
const CONNECTOR_TOP = (TRACK_HEIGHT - 2) / 2;

function segmentTrackStyle(fromStepIndex: number): {
  left: DimensionValue;
  width: DimensionValue;
} {
  return {
    left: `${((fromStepIndex + 0.5) / STEP_COUNT) * 100}%` as DimensionValue,
    width: `${(1 / STEP_COUNT) * 100}%` as DimensionValue,
  };
}

type Props = {
  mission: DriverMission;
  etaSnapshot?: DriverEtaSnapshot | null;
  remainingDistanceKm?: number | null;
};

function ApproachSegmentBar({
  fromStepIndex,
  progress,
}: {
  fromStepIndex: number;
  progress: number;
}) {
  const widthAnim = useRef(new Animated.Value(0)).current;
  const trackStyle = segmentTrackStyle(fromStepIndex);

  useEffect(() => {
    Animated.timing(widthAnim, {
      toValue: Math.min(1, Math.max(0, progress)),
      duration: 520,
      easing: Easing.out(Easing.cubic),
      useNativeDriver: false,
    }).start();
  }, [progress, widthAnim]);

  const fillWidth = widthAnim.interpolate({
    inputRange: [0, 1],
    outputRange: ["0%", "100%"],
  });

  return (
    <View
      style={[styles.approachTrack, trackStyle]}
      accessibilityElementsHidden
    >
      <Animated.View style={[styles.approachFill, { width: fillWidth }]} />
    </View>
  );
}

export function MissionProgressStepper({ mission, etaSnapshot, remainingDistanceKm }: Props) {
  const progress = resolveMissionStepperProgress(mission);
  const approach = useMissionStepperApproachProgress(mission, {
    etaSnapshot,
    remainingDistanceKm,
  });

  return (
    <View style={styles.wrap} accessibilityLabel="Progression de la mission">
      {approach ? (
        <ApproachSegmentBar fromStepIndex={approach.fromStepIndex} progress={approach.progress} />
      ) : null}
      <View style={styles.columnsRow}>
        {STEPS.map((label, index) => {
          const isComplete = index < progress.completedCount;
          const isActive = progress.activeIndex === index && !isComplete;
          const isLast = index === STEPS.length - 1;
          const isApproachTarget =
            approach != null && index === approach.targetStepIndex && !isComplete;

          const leftComplete =
            index > 0 && isMissionStepperSegmentComplete(index - 1, progress);
          const rightComplete =
            !isLast && isMissionStepperSegmentComplete(index, progress);

          return (
            <View key={label} style={styles.stepColumn}>
              <View style={styles.nodeRail}>
                {index > 0 ? (
                  <View
                    style={[
                      styles.connector,
                      styles.connectorLeft,
                      leftComplete ? styles.connectorComplete : styles.connectorPending,
                    ]}
                    accessibilityElementsHidden
                  />
                ) : null}
                {!isLast ? (
                  <View
                    style={[
                      styles.connector,
                      styles.connectorRight,
                      rightComplete ? styles.connectorComplete : styles.connectorPending,
                    ]}
                    accessibilityElementsHidden
                  />
                ) : null}
                <View
                  style={[
                    styles.node,
                    isComplete && styles.nodeComplete,
                    isActive && styles.nodeActive,
                    isApproachTarget && styles.nodeApproachTarget,
                    !isComplete && !isActive && !isApproachTarget && styles.nodePending,
                  ]}
                  accessibilityElementsHidden
                >
                  {isComplete ? <Ionicons name="checkmark" size={9} color="#FFFFFF" /> : null}
                  {isActive ? <View style={styles.nodeActiveDot} /> : null}
                </View>
              </View>
              <AppText
                variant="caption"
                style={[
                  styles.stepLabel,
                  (isComplete || isActive || isApproachTarget) && styles.stepLabelActive,
                  !isComplete && !isActive && !isApproachTarget && styles.stepLabelMuted,
                ]}
                numberOfLines={2}
              >
                {label}
              </AppText>
            </View>
          );
        })}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    paddingVertical: 1,
    position: "relative",
  },
  approachTrack: {
    position: "absolute",
    top: CONNECTOR_TOP,
    height: 3,
    borderRadius: 2,
    backgroundColor: D.stepLine,
    zIndex: 3,
    overflow: "hidden",
  },
  approachFill: {
    height: "100%",
    backgroundColor: D.brand,
    borderRadius: 2,
  },
  columnsRow: {
    flexDirection: "row",
    alignItems: "flex-start",
  },
  stepColumn: {
    flex: 1,
    minWidth: 0,
    alignItems: "center",
    gap: 4,
  },
  nodeRail: {
    width: "100%",
    height: TRACK_HEIGHT,
    alignItems: "center",
    justifyContent: "center",
    position: "relative",
  },
  node: {
    width: NODE_SIZE,
    height: NODE_SIZE,
    borderRadius: NODE_SIZE / 2,
    alignItems: "center",
    justifyContent: "center",
    zIndex: 4,
  },
  nodeComplete: {
    backgroundColor: D.brand,
    borderWidth: 0,
  },
  nodeActive: {
    backgroundColor: "#FFFFFF",
    borderWidth: 2,
    borderColor: D.brand,
  },
  nodeApproachTarget: {
    backgroundColor: "#FFFFFF",
    borderWidth: 2,
    borderColor: "rgba(0, 121, 107, 0.5)",
  },
  nodePending: {
    backgroundColor: "#FFFFFF",
    borderWidth: 2,
    borderColor: D.stepLine,
  },
  nodeActiveDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: D.brand,
  },
  connector: {
    position: "absolute",
    top: CONNECTOR_TOP,
    height: 2,
    zIndex: 1,
  },
  connectorLeft: {
    left: 0,
    right: "50%",
  },
  connectorRight: {
    left: "50%",
    right: 0,
  },
  connectorComplete: {
    backgroundColor: D.brand,
  },
  connectorPending: {
    backgroundColor: D.stepLine,
  },
  stepLabel: {
    width: "100%",
    fontSize: FONT_SIZE.px8,
    lineHeight: 10,
    textAlign: "center",
    fontWeight: "600",
    paddingHorizontal: 1,
  },
  stepLabelActive: {
    color: D.brand,
    fontWeight: "800",
  },
  stepLabelMuted: {
    color: D.textMuted,
    fontWeight: "500",
  },
});
