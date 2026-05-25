import { memo, useCallback, useEffect, useMemo, useState } from "react";
import { Marker } from "react-native-maps";
import { Platform } from "react-native";

import type { FleetDriverMapItem } from "./fleetMapTypes";
import { FleetMapRasterMarker } from "./FleetMapRasterMarker";
import { FleetDriverLivePulse } from "./FleetDriverLivePulse";
import { shouldFleetMarkerLivePulse } from "./fleetMapLiveMarker";
import { FLEET_STATUS_THEME } from "./mapStatusTheme";
import { driverFleetMarkerTitle } from "../../utils/companyDriverMapStatus";
import { buildFleetDriverMarkerImageSource } from "./fleetNativeMarkerImage";
import { countDriverMarkerRender } from "./fleetMapDevInstrumentation";
import { recordDriverMarkerRender } from "../../../../core/observability/perfInstrumentation";

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

  const coordinate = useMemo(
    () => ({ latitude: item.latitude, longitude: item.longitude }),
    [item.latitude, item.longitude]
  );

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
  }, [showLivePulse, coordinate.latitude, coordinate.longitude]);

  const handlePress = useCallback(
    (e: { stopPropagation?: () => void }) => {
      e?.stopPropagation?.();
      onPress?.(item.driver_id);
    },
    [item.driver_id, onPress]
  );

  void vectorMode;

  return (
    <>
      {showLivePulse ? (
        <Marker
          coordinate={coordinate}
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
        coordinate={coordinate}
        imageSource={imageSource}
        anchor={{ x: 0.5, y: 1 }}
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
    prev.selected === next.selected &&
    prev.dimmed === next.dimmed &&
    prev.vectorMode === next.vectorMode &&
    prev.item.enrichment.operationalStatus === next.item.enrichment.operationalStatus &&
    prev.onPress === next.onPress
  );
}

export const DriverMarker = memo(DriverMarkerComponent, areDriverMarkerPropsEqual);
