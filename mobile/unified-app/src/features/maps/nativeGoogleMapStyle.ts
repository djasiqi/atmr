import {
  LIRIE_MAP_STYLES,
  LIRIE_POI_SUPPRESSION_STYLES,
  resolveExpoLirieGoogleMapLayer,
} from "./expoLirieGoogleMapLayer";

/** Comportement carte opérationnelle — masque POI / chrome Google natif (cockpit mobile). */
export function getNativeOperationalMapBehaviorProps(): {
  showsPointsOfInterest: false;
  showsIndoors: false;
  showsBuildings: false;
  showsCompass: false;
  showsScale: false;
  toolbarEnabled: false;
  moveOnMarkerPress: false;
} {
  return {
    showsPointsOfInterest: false,
    showsIndoors: false,
    showsBuildings: false,
    showsCompass: false,
    showsScale: false,
    toolbarEnabled: false,
    moveOnMarkerPress: false,
  };
}

export type NativeGoogleMapViewProps = ReturnType<typeof getNativeOperationalMapBehaviorProps> & {
  customMapStyle?: object[];
  googleMapId?: string;
};

/**
 * Props carte pour `react-native-maps` (iOS / Android) : style Lirie + calme opérationnel.
 * Variables : `EXPO_PUBLIC_GOOGLE_MAPS_LIRIE_STYLE`, `EXPO_PUBLIC_GOOGLE_MAPS_MAP_ID`.
 */
export function getNativeGoogleMapViewStyleProps(): NativeGoogleMapViewProps {
  const operational = getNativeOperationalMapBehaviorProps();
  const layer = resolveExpoLirieGoogleMapLayer();
  if (layer.kind === "cloud") {
    return {
      googleMapId: layer.mapId,
      customMapStyle: [...LIRIE_POI_SUPPRESSION_STYLES] as object[],
      ...operational,
    };
  }
  return { customMapStyle: [...LIRIE_MAP_STYLES] as object[], ...operational };
}
