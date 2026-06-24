import { memo, useCallback, useEffect, useMemo } from "react";

import type { FleetDriverMapItem } from "./fleetMapTypes";
import { FleetMapRasterMarker } from "./FleetMapRasterMarker";
import { FLEET_STATUS_THEME } from "./mapStatusTheme";
import { driverFleetMarkerTitle } from "../../utils/companyDriverMapStatus";
import { buildFleetDriverMarkerImageSource } from "./fleetNativeMarkerImage";
import { resolveFleetMarkerAnchor } from "./resolveFleetMarkerAnchor";
import { countDriverMarkerRender } from "./fleetMapDevInstrumentation";
import { recordDriverMarkerRender } from "../../../../core/observability/perfInstrumentation";
import { isValidMapCoord } from "./mapsIosNewArchSafeMode";
import { useFleetMarkerMotion } from "./useFleetMarkerMotion";
import { isFleetDriverMarkerStale } from "./fleetMapStale";

type Props = {
  item: FleetDriverMapItem;
  selected?: boolean;
  dimmed?: boolean;
  vectorMode?: boolean;
  onPress?: (driverId: number) => void;
};

function DriverMarkerComponent({
  item,
  selected = false,
  dimmed = false,
  vectorMode = false,
  onPress,
}: Props) {
  const status = item.enrichment.operationalStatus;
  const theme = FLEET_STATUS_THEME[status];
  const isStale = useMemo(() => isFleetDriverMarkerStale(item), [item]);

  const imageSource = useMemo(
    () => buildFleetDriverMarkerImageSource(status, item, { isStale }),
    [isStale, item, status]
  );
  const markerAnchor = useMemo(
    () => resolveFleetMarkerAnchor(imageSource),
    [imageSource]
  );

  const targetCoordinate = useMemo(
    () => ({ latitude: item.latitude, longitude: item.longitude }),
    [item.latitude, item.longitude]
  );

  const { displayCoordinate, primaryMarkerRef } = useFleetMarkerMotion({
    target: targetCoordinate,
    markerKey: String(item.driver_id),
    recordedAt: item.recorded_at ?? item.timestamp,
    locationStatus: item.location_status,
  });

  useEffect(() => {
    countDriverMarkerRender(item.driver_id);
    recordDriverMarkerRender();
  });

  const handlePress = useCallback(
    (e: { stopPropagation?: () => void }) => {
      e?.stopPropagation?.();
      onPress?.(item.driver_id);
    },
    [item.driver_id, onPress]
  );

  void vectorMode;
  void selected;

  if (!imageSource.uri?.trim()) {
    return null;
  }

  if (!isValidMapCoord(displayCoordinate.latitude, displayCoordinate.longitude)) {
    return null;
  }

  return (
    <FleetMapRasterMarker
      ref={primaryMarkerRef}
      coordinate={displayCoordinate}
      imageSource={imageSource}
      anchor={markerAnchor}
      title={driverFleetMarkerTitle(item)}
      onPress={handlePress}
      zIndex={theme.priority}
      opacity={dimmed ? 0.45 : 1}
    />
  );
}

function areDriverMarkerPropsEqual(prev: Props, next: Props): boolean {
  return (
    prev.item.driver_id === next.item.driver_id &&
    prev.item.latitude === next.item.latitude &&
    prev.item.longitude === next.item.longitude &&
    prev.item.location_status === next.item.location_status &&
    prev.item.last_seen_seconds === next.item.last_seen_seconds &&
    prev.item.tracking_display_status === next.item.tracking_display_status &&
    prev.item.recorded_at === next.item.recorded_at &&
    prev.item.timestamp === next.item.timestamp &&
    prev.selected === next.selected &&
    prev.dimmed === next.dimmed &&
    prev.vectorMode === next.vectorMode &&
    prev.item.enrichment.operationalStatus === next.item.enrichment.operationalStatus &&
    prev.onPress === next.onPress
  );
}

export const DriverMarker = memo(DriverMarkerComponent, areDriverMarkerPropsEqual);
