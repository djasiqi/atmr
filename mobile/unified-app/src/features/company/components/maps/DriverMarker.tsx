import { memo, useCallback, useEffect, useMemo, useState } from "react";
import { Platform } from "react-native";
import { Marker } from "react-native-maps";

import type { FleetDriverMapItem } from "./fleetMapTypes";
import { FleetMapRasterMarker } from "./FleetMapRasterMarker";
import { FleetDriverMarkerVisual } from "./FleetDriverMarkerVisual";
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

function useAndroidMarkerTracksViewChanges(visualKey: string): boolean {
  const [tracksViewChanges, setTracksViewChanges] = useState(Platform.OS === "android");
  useEffect(() => {
    if (Platform.OS !== "android") return;
    setTracksViewChanges(true);
    const timer = setTimeout(() => setTracksViewChanges(false), 150);
    return () => clearTimeout(timer);
  }, [visualKey]);
  return tracksViewChanges;
}

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

  const androidVisualKey = `${status}:${isStale}:${dimmed}:${selected}:${item.driver_id}:${item.first_name ?? ""}:${item.last_name ?? ""}:${item.full_name ?? ""}`;
  const androidTracksViewChanges = useAndroidMarkerTracksViewChanges(androidVisualKey);

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

  if (!isValidMapCoord(displayCoordinate.latitude, displayCoordinate.longitude)) {
    return null;
  }

  if (Platform.OS === "android") {
    return (
      <Marker
        ref={primaryMarkerRef}
        coordinate={displayCoordinate}
        anchor={{ x: 0.5, y: 0.5 }}
        tracksViewChanges={androidTracksViewChanges}
        zIndex={theme.priority}
        opacity={dimmed ? 0.45 : 1}
        title={driverFleetMarkerTitle(item)}
        onPress={handlePress}
      >
        <FleetDriverMarkerVisual
          status={status}
          selected={selected}
          dimmed={false}
          driver={item}
          compactForMap
        />
      </Marker>
    );
  }

  if (!imageSource.uri?.trim()) {
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
    prev.item.driver_name === next.item.driver_name &&
    prev.item.first_name === next.item.first_name &&
    prev.item.last_name === next.item.last_name &&
    prev.item.full_name === next.item.full_name &&
    prev.selected === next.selected &&
    prev.dimmed === next.dimmed &&
    prev.vectorMode === next.vectorMode &&
    prev.item.enrichment.operationalStatus === next.item.enrichment.operationalStatus &&
    prev.onPress === next.onPress
  );
}

export const DriverMarker = memo(DriverMarkerComponent, areDriverMarkerPropsEqual);
