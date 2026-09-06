import { useCallback, useEffect, useMemo, useState } from "react";
import { StyleSheet, View } from "react-native";
import type { CompanyDriverLiveLocation } from "../../api/contracts";
import type { CockpitLiveStatus } from "../../dashboard/cockpit/cockpitTypes";
import type { CompanyDataFreshness } from "../../realtime/companyRealtimeState";
import {
  buildLiveCoverageRows,
  computeGpsCoverageCounts,
  formatGpsCoverageSummary,
} from "./liveGpsCoverage";
import { LiveCoverageSheet } from "./LiveCoverageSheet";
import { LiveSystemStatusPill } from "./LiveSystemStatusPill";

type Props = {
  drivers: CompanyDriverLiveLocation[];
  rosterResolved: boolean;
  visualWorkEnabled: boolean;
  variant: "header" | "default";
  status: CockpitLiveStatus;
  realtimeSocketExpected: boolean;
  activityHint?: string | null;
  dataFreshness?: CompanyDataFreshness;
  animationIntensity: number;
  onOpen?: () => void;
  onSelectDriver: (driverId: number) => void;
  bottomOffset?: number;
};

/**
 * Horloge N/T + feuille couverture isolées du reste du cockpit.
 * Un tick 5 s ne rerender plus la carte / les courses / la tab bar.
 */
export function CockpitLiveCoverageIsland({
  drivers,
  rosterResolved,
  visualWorkEnabled,
  variant,
  status,
  realtimeSocketExpected,
  activityHint,
  dataFreshness,
  animationIntensity,
  onOpen,
  onSelectDriver,
  bottomOffset,
}: Props) {
  const [coverageNowMs, setCoverageNowMs] = useState(() => Date.now());
  const [liveCoverageOpen, setLiveCoverageOpen] = useState(false);

  useEffect(() => {
    if (!visualWorkEnabled) return;
    setCoverageNowMs(Date.now());
    const id = setInterval(() => setCoverageNowMs(Date.now()), 5_000);
    return () => clearInterval(id);
  }, [visualWorkEnabled]);

  const gpsCoverage = useMemo(
    () => computeGpsCoverageCounts(drivers, coverageNowMs),
    [drivers, coverageNowMs]
  );
  const liveCoverageRows = useMemo(
    () =>
      liveCoverageOpen ? buildLiveCoverageRows(drivers, coverageNowMs) : [],
    [drivers, coverageNowMs, liveCoverageOpen]
  );
  const liveCoverageSummary = useMemo(
    () => formatGpsCoverageSummary(gpsCoverage.liveCount, gpsCoverage.totalCount),
    [gpsCoverage.liveCount, gpsCoverage.totalCount]
  );

  const onLivePillPress = useCallback(() => {
    onOpen?.();
    setLiveCoverageOpen(true);
  }, [onOpen]);

  const pill = (
    <LiveSystemStatusPill
      variant={variant === "header" ? "header" : "default"}
      status={status}
      realtimeSocketExpected={realtimeSocketExpected}
      activityHint={activityHint}
      gpsCoverage={gpsCoverage}
      gpsCoverageResolved={rosterResolved}
      dataFreshness={dataFreshness}
      onPress={onLivePillPress}
      animationIntensity={visualWorkEnabled ? animationIntensity : 0}
    />
  );

  const sheet = (
    <LiveCoverageSheet
      visible={liveCoverageOpen}
      onClose={() => setLiveCoverageOpen(false)}
      summary={liveCoverageSummary}
      rows={liveCoverageRows}
      onSelectDriver={onSelectDriver}
    />
  );

  if (variant === "header") {
    return (
      <>
        {pill}
        {sheet}
      </>
    );
  }

  return (
    <>
      <View
        style={[s.livePillWrap, bottomOffset != null ? { bottom: bottomOffset } : null]}
        pointerEvents="box-none"
      >
        {pill}
      </View>
      {sheet}
    </>
  );
}

const s = StyleSheet.create({
  livePillWrap: {
    position: "absolute",
    left: 0,
    right: 0,
    alignItems: "center",
    zIndex: 34,
  },
});
