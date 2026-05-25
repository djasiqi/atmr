import { useEffect } from "react";

import { StyleSheet, View } from "react-native";

import { GoogleMapsFleetCanvas } from "./maps/GoogleMapsFleetCanvas.web";

import type { EnterpriseDriversMapProps } from "./EnterpriseDriversMap";

import {

  LIRIE_GOOGLE_MAP_ATTRIBUTION_HIDE_CSS,

  LIRIE_GOOGLE_MAP_ATTRIBUTION_MASK_COLOR,

  LIRIE_GOOGLE_MAP_LOGO_CLIP_PX,

} from "../../maps/lirieMapChrome";



const LIRIE_MAP_CHROME_STYLE_ID = "lirie-google-map-chrome-hide";



function useHideGoogleMapDefaultChrome() {

  useEffect(() => {

    if (typeof document === "undefined") return;

    if (document.getElementById(LIRIE_MAP_CHROME_STYLE_ID)) return;

    const style = document.createElement("style");

    style.id = LIRIE_MAP_CHROME_STYLE_ID;

    style.textContent = LIRIE_GOOGLE_MAP_ATTRIBUTION_HIDE_CSS;

    document.head.appendChild(style);

  }, []);

}



const BORDER = "rgba(145, 165, 157, 0.45)";



const styles = StyleSheet.create({

  root: {

    flex: 1,

    width: "100%",

    height: "100%",

    backgroundColor: "#FFFFFF",

    borderWidth: 1,

    borderColor: BORDER,

    borderRadius: 12,

    overflow: "hidden",

    boxShadow: "0 2px 10px rgba(22, 58, 52, 0.06)",

  },

  cockpitRoot: {

    borderWidth: 0,

    borderRadius: 0,

    backgroundColor: "#E8EDEB",

    boxShadow: "none",

  },

  clipWrap: {

    flex: 1,

    width: "100%",

    height: "100%",

    overflow: "hidden",

    position: "relative",

  },

  attributionMask: {

    position: "absolute",

    left: 0,

    right: 0,

    bottom: 0,

    zIndex: 10,

  },

});



export function EnterpriseDriversMap({

  drivers,

  markers,

  mapHeight = 200,

  containerStyle,

  selectedDriverId,

  layers,

  activeRoute,

  missionOverlays,

  missionsById,

  stableEtaAnchors,

  selectedMissionId,

  heatmapPoints,

  recenterRegion,

  recenterMode,

  recenterToken,

  cameraInsets,

  onDriverPress,

  onClusterPress,

  pinnedClusterFocus,

  logoClipFill = false,

  mapAttributionMaskColor = LIRIE_GOOGLE_MAP_ATTRIBUTION_MASK_COLOR,

}: EnterpriseDriversMapProps) {

  useHideGoogleMapDefaultChrome();

  const maskColor = mapAttributionMaskColor;

  const maskHeight = LIRIE_GOOGLE_MAP_LOGO_CLIP_PX + 8;



  return (

    <View

      style={[styles.root, logoClipFill ? styles.cockpitRoot : null, containerStyle]}

      className="liri-web-map-showcase"

      accessibilityLabel="Carte des chauffeurs en direct"

    >

      <View style={styles.clipWrap}>

        <GoogleMapsFleetCanvas

          drivers={drivers}

          markers={markers}

          height={mapHeight}

          selectedDriverId={selectedDriverId}

          layers={layers}

          activeRoute={activeRoute}

          missionOverlays={missionOverlays}

          missionsById={missionsById}

          stableEtaAnchors={stableEtaAnchors}

          selectedMissionId={selectedMissionId}

          heatmapPoints={heatmapPoints}

          recenterRegion={recenterRegion}

          recenterMode={recenterMode}

          recenterToken={recenterToken}

          cameraInsets={cameraInsets}

          onDriverPress={onDriverPress}

          onClusterPress={onClusterPress}

          pinnedClusterFocus={pinnedClusterFocus}

        />

        <View

          pointerEvents="none"

          style={[

            styles.attributionMask,

            { height: maskHeight, backgroundColor: maskColor },

          ]}

        />

      </View>

    </View>

  );

}

