import { StyleSheet, View } from "react-native";
import { AppText } from "../../../../design/ui/AppText";
import type { CockpitOrchestrationDecision } from "../../dashboard/cockpit/cockpitOrchestrator";
import type { CockpitUiState } from "../../dashboard/cockpit/cockpitTypes";
import { getCockpitMetricSnapshot } from "../../dashboard/cockpit/cockpitMetrics";
import { FONT_SIZE } from "../../../../design/responsive/typographyTokens";

type Props = {
  uiState: CockpitUiState;
  orchestration?: CockpitOrchestrationDecision;
};

/**
 * Panneau debug — uniquement __DEV__ + EXPO_PUBLIC_COCKPIT_DEBUGGER=1 (via cockpit.flags).
 */
export function CockpitStateDebugger({ uiState, orchestration }: Props) {
  if (!__DEV__) return null;
  if (process.env.EXPO_PUBLIC_COCKPIT_DEBUGGER !== "1") return null;

  const metrics = getCockpitMetricSnapshot();
  const activeOverlays = Object.entries(uiState.overlays)
    .filter(([, v]) => v)
    .map(([k]) => k)
    .join(", ");

  return (
    <View style={s.box} pointerEvents="none">
      <AppText variant="caption" style={s.line}>
        mode={uiState.mode} · fsm={orchestration?.fsmState ?? "—"} · intent=
        {orchestration?.mapIntent ?? "—"}
      </AppText>
      <AppText variant="caption" style={s.line}>
        attention={orchestration?.attentionLevel ?? "—"} · density=
        {orchestration?.density.level ?? "—"} · motion={orchestration?.motionPolicy ?? "—"}
      </AppText>
      <AppText variant="caption" style={s.line}>
        routes={orchestration?.maxVisibleRoutes ?? "—"} · camera=
        {orchestration?.cameraPolicy ?? "—"} · reduced=
        {orchestration?.reducedComplexity ? "1" : "0"}
      </AppText>
      <AppText variant="caption" style={s.line}>
        health={orchestration?.fleetHealthScore ?? "—"} · vector=
        {orchestration?.globalVectorMode ? "1" : "0"}
      </AppText>
      <AppText variant="caption" style={s.line} numberOfLines={2}>
        overlays: {activeOverlays || "—"}
      </AppText>
      <AppText variant="caption" style={s.line}>
        reason={uiState.lastTransitionReason ?? "—"}
      </AppText>
      {Object.keys(metrics).length > 0 ? (
        <AppText variant="caption" style={s.line} numberOfLines={2}>
          metrics: {JSON.stringify(metrics)}
        </AppText>
      ) : null}
    </View>
  );
}

const s = StyleSheet.create({
  box: {
    position: "absolute",
    top: 120,
    left: 8,
    right: 8,
    zIndex: 99,
    backgroundColor: "rgba(15, 23, 42, 0.82)",
    borderRadius: 8,
    padding: 8,
    gap: 2,
  },
  line: {
    color: "#E2E8F0",
    fontSize: FONT_SIZE.px9,
    fontFamily: "monospace",
  },
});
