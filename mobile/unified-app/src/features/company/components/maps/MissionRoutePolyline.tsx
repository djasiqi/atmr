import { useEffect, useMemo, useRef, useState } from "react";

import { Animated } from "react-native";

import { Polyline } from "react-native-maps";



import type { FleetMapLatLng, FleetRoutedPathState } from "./fleetMapDirections";

import { FLEET_MISSION_MAP_POLICY } from "./fleetMapMissionPolicies";

import {

  FLEET_NATIVE_ROUTE_RENDER,

  isAndroidFleetMapPlatform,

  isNativeFleetMapPlatform,

  prepareFleetRouteCoordsForNativeRender,

  resolveNativeMissionRouteStrokes,

} from "./fleetMapNativeRouteRender";

import {

  resolveFleetMissionRouteLegRenders,

  type FleetMissionOverlay,

  type FleetMissionRouteLegRender,

} from "./fleetMapMissionVisual";



type Props = {

  overlay: FleetMissionOverlay;

  routedPathsByMissionId: ReadonlyMap<string, FleetMapLatLng[]>;

  routedStateByMissionId: ReadonlyMap<string, FleetRoutedPathState>;

};



function applyStrokeOpacity(color: string, alpha: number): string {

  const clamped = Math.max(0, Math.min(1, alpha));

  if (color.startsWith("rgba")) {

    return color.replace(/[\d.]+\)$/, `${clamped})`);

  }

  if (color.startsWith("#") && color.length === 7) {

    const a = Math.round(clamped * 255)

      .toString(16)

      .padStart(2, "0");

    return `${color}${a}`;

  }

  return color;

}



function withAlpha(color: string, alpha: number): string {

  return applyStrokeOpacity(color, alpha);

}



function MissionRouteStrokePolylines({

  overlay,

  legRender,

}: {

  overlay: FleetMissionOverlay;

  legRender: FleetMissionRouteLegRender;

}) {

  const target = overlay.displayOpacity * legRender.style.opacity;

  const [opacity, setOpacity] = useState(target * 0.35);

  const animRef = useRef(new Animated.Value(target * 0.35)).current;



  useEffect(() => {

    const duration = overlay.isSelected

      ? FLEET_MISSION_MAP_POLICY.routeEnterMs

      : FLEET_MISSION_MAP_POLICY.routeExitMs;

    Animated.timing(animRef, {

      toValue: target,

      duration,

      useNativeDriver: false,

    }).start();

    const id = animRef.addListener(({ value }) => setOpacity(value));

    return () => animRef.removeListener(id);

  }, [animRef, overlay.isSelected, overlay.missionId, target, legRender.leg]);



  const renderCoords = useMemo(

    () =>

      prepareFleetRouteCoordsForNativeRender(

        legRender.coordinates.map((point) => ({

          latitude: point.latitude,

          longitude: point.longitude,

        }))

      ),

    [legRender.coordinates]

  );



  if (renderCoords.length < 2) return null;



  const routeStyle = legRender.style;
  if (!routeStyle) return null;

  const nativeStrokes = isNativeFleetMapPlatform()

    ? resolveNativeMissionRouteStrokes(routeStyle)

    : { mainStroke: routeStyle.strokeWidth, glowStroke: routeStyle.glowWidth };

  const isAndroidNative = isAndroidFleetMapPlatform();

  const glowOpacity =

    opacity *

    (isNativeFleetMapPlatform()

      ? isAndroidNative

        ? 0.22

        : FLEET_NATIVE_ROUTE_RENDER.glowOpacityScale

      : 0.85);

  const lineOpacity = Math.min(

    isAndroidNative ? 0.78 : FLEET_NATIVE_ROUTE_RENDER.maxMainLineOpacity,

    opacity

  );

  const showGlowPolyline = !isAndroidNative || overlay.isSelected;



  return (

    <>

      {showGlowPolyline ? (

        <Polyline

          key={`glow-${legRender.leg}`}

          coordinates={renderCoords}

          strokeColor={withAlpha(routeStyle.glowColor, glowOpacity)}

          strokeWidth={nativeStrokes.glowStroke}

          lineCap="round"

          lineJoin="round"

          lineDashPattern={routeStyle.lineDashPattern ?? undefined}

          zIndex={legRender.zIndex}

        />

      ) : null}

      <Polyline

        key={`line-${legRender.leg}`}

        coordinates={renderCoords}

        strokeColor={applyStrokeOpacity(routeStyle.color, lineOpacity)}

        strokeWidth={nativeStrokes.mainStroke}

        lineCap="round"

        lineJoin="round"

        lineDashPattern={routeStyle.lineDashPattern ?? undefined}

        zIndex={legRender.zIndex + 1}

      />

    </>

  );

}



export function MissionRoutePolyline({

  overlay,

  routedPathsByMissionId,

  routedStateByMissionId,

}: Props) {

  const legRenders = resolveFleetMissionRouteLegRenders(

    overlay,

    routedPathsByMissionId,

    routedStateByMissionId

  );



  return (

    <>

      {legRenders.map((legRender) => (

        <MissionRouteStrokePolylines

          key={`${overlay.missionId}-${legRender.leg}`}

          overlay={overlay}

          legRender={legRender}

        />

      ))}

    </>

  );

}



export function MissionRoutePolylines({

  overlays,

  routedPathsByMissionId,

  routedStateByMissionId,

}: {

  overlays: FleetMissionOverlay[];

  routedPathsByMissionId: ReadonlyMap<string, FleetMapLatLng[]>;

  routedStateByMissionId: ReadonlyMap<string, FleetRoutedPathState>;

}) {

  return (

    <>

      {overlays.map((overlay) => (

        <MissionRoutePolyline

          key={`mission-route-${overlay.missionId}`}

          overlay={overlay}

          routedPathsByMissionId={routedPathsByMissionId}

          routedStateByMissionId={routedStateByMissionId}

        />

      ))}

    </>

  );

}

