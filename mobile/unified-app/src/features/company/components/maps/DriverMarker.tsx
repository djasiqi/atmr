import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Marker } from "react-native-maps";
import { Platform } from "react-native";

import type { FleetDriverMapItem } from "./fleetMapTypes";
import { FleetMapRasterMarker } from "./FleetMapRasterMarker";
import { FleetDriverLivePulse } from "./FleetDriverLivePulse";
import { shouldFleetMarkerLivePulse } from "./fleetMapLiveMarker";
import { FLEET_STATUS_THEME } from "./mapStatusTheme";
import { driverFleetMarkerTitle } from "../../utils/companyDriverMapStatus";
import { buildFleetDriverMarkerImageSource } from "./fleetNativeMarkerImage";
import { resolveFleetMarkerAnchor } from "./resolveFleetMarkerAnchor";
import { countDriverMarkerRender } from "./fleetMapDevInstrumentation";
import { recordDriverMarkerRender } from "../../../../core/observability/perfInstrumentation";
import { IOS_MAP_NO_CUSTOM_MARKER_CHILDREN, isValidMapCoord } from "./mapsIosNewArchSafeMode";
import { useFleetMarkerMotion } from "./useFleetMarkerMotion";

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
  const showLivePulse = useMemo(
    () => shouldFleetMarkerLivePulse(status, item),
    [item, status]
  );
  const pulseVariant = status === "on_mission" ? "mission" : "available";

  const imageSource = useMemo(
    () => buildFleetDriverMarkerImageSource(status, selected),
    [selected, status]
  );
  const markerAnchor = useMemo(
    () => resolveFleetMarkerAnchor(imageSource),
    [imageSource]
  );

  const targetCoordinate = useMemo(
    () => ({ latitude: item.latitude, longitude: item.longitude }),
    [item.latitude, item.longitude]
  );

  const pulseMarkerRef = useRef<Marker | null>(null);
  const { displayCoordinate, primaryMarkerRef } = useFleetMarkerMotion({
    target: targetCoordinate,
    markerKey: String(item.driver_id),
    recordedAt: item.recorded_at ?? item.timestamp,
    locationStatus: item.location_status,
    secondaryMarkerRef: pulseMarkerRef,
  });

  const [pulseTracksViewChanges, setPulseTracksViewChanges] = useState(Platform.OS === "android");

  useEffect(() => {
    countDriverMarkerRender(item.driver_id);
    recordDriverMarkerRender();
  });

  useEffect(() => {
    if (!showLivePulse) return;
    setPulseTracksViewChanges(Platform.OS === "android");
    const id = setTimeout(() => setPulseTracksViewChanges(false), 400);
    return () => clearTimeout(id);
  }, [showLivePulse, displayCoordinate.latitude, displayCoordinate.longitude]);

  const handlePress = useCallback(
    (e: { stopPropagation?: () => void }) => {
      e?.stopPropagation?.();
      onPress?.(item.driver_id);
    },
    [item.driver_id, onPress]
  );

  void vectorMode;

  if (!imageSource.uri?.trim()) {
    return null;
  }

  if (!isValidMapCoord(displayCoordinate.latitude, displayCoordinate.longitude)) {
    return null;
  }

  return (
    <>
      {showLivePulse && !IOS_MAP_NO_CUSTOM_MARKER_CHILDREN ? (
        <Marker
          ref={pulseMarkerRef}
          coordinate={displayCoordinate}
          anchor={{ x: 0.5, y: 0.58 }}
          tracksViewChanges={pulseTracksViewChanges}
          zIndex={(selected ? 999 : theme.priority) - 1}
          opacity={dimmed ? 0.45 : 1}
          pointerEvents="none"
        >
          <FleetDriverLivePulse color={theme.fill} variant={pulseVariant} />
        </Marker>
      ) : null}
      <FleetMapRasterMarker
        ref={primaryMarkerRef}
        coordinate={displayCoordinate}
        imageSource={imageSource}
        anchor={markerAnchor}
        title={driverFleetMarkerTitle(item)}
        onPress={handlePress}
        zIndex={selected ? 999 : theme.priority}
        opacity={dimmed ? 0.45 : 1}
      />
    </>
  );
}

function areDriverMarkerPropsEqual(prev: Props, next: Props): boolean {
  return (
    prev.item.driver_id === next.item.driver_id &&
    prev.item.latitude === next.item.latitude &&
    prev.item.longitude === next.item.longitude &&
    prev.item.location_status === next.item.location_status &&
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
