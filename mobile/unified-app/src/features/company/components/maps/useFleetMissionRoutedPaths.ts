import { useEffect, useMemo, useState } from "react";

import {

  buildMissionDirectionsPlanSignature,

  fleetRoutePathKey,

  type FleetMapLatLng,

  type FleetRoutedPathState,

} from "./fleetMapDirections";

import { fetchFleetDirectionsPathsForOverlaysNative } from "./fleetMapDirectionsNative";

import type { FleetMissionOverlay } from "./fleetMapMissionVisual";



export function useFleetMissionRoutedPaths(

  overlays: FleetMissionOverlay[],

  apiKey: string | undefined

): {

  routedPathsByMissionId: Map<string, FleetMapLatLng[]>;

  routedStateByMissionId: Map<string, FleetRoutedPathState>;

} {

  const planSignature = useMemo(

    () => (overlays.length > 0 ? buildMissionDirectionsPlanSignature(overlays) : ""),

    [overlays]

  );



  const [routedPathsByMissionId, setRoutedPathsByMissionId] = useState(

    () => new Map<string, FleetMapLatLng[]>()

  );

  const [routedStateByMissionId, setRoutedStateByMissionId] = useState(

    () => new Map<string, FleetRoutedPathState>()

  );



  useEffect(() => {

    if (!apiKey || overlays.length === 0) {

      setRoutedPathsByMissionId(new Map());

      setRoutedStateByMissionId(new Map());

      return;

    }



    let cancelled = false;

    const loadingStates = new Map<string, FleetRoutedPathState>();

    for (const overlay of overlays) {

      if (overlay.legDirectionsPlans) {

        for (const leg of ["to_pickup", "to_dropoff"] as const) {

          if (overlay.legDirectionsPlans[leg]) {

            loadingStates.set(fleetRoutePathKey(overlay.missionId, leg), "loading");

          }

        }

      } else if (overlay.directionsPlan) {

        loadingStates.set(fleetRoutePathKey(overlay.missionId), "loading");

      }

    }

    setRoutedStateByMissionId(loadingStates);



    void (async () => {

      const fetched = await fetchFleetDirectionsPathsForOverlaysNative(overlays, apiKey);

      if (cancelled) return;



      const nextPaths = new Map<string, FleetMapLatLng[]>();

      const nextStates = new Map<string, FleetRoutedPathState>();

      for (const overlay of overlays) {

        if (overlay.legDirectionsPlans) {

          for (const leg of ["to_pickup", "to_dropoff"] as const) {

            if (!overlay.legDirectionsPlans[leg]) continue;

            const key = fleetRoutePathKey(overlay.missionId, leg);

            const path = fetched.get(key);

            if (path && path.length >= 2) {

              nextPaths.set(key, path);

              nextStates.set(key, "ready");

            } else {

              nextStates.set(key, "failed");

            }

          }

          continue;

        }

        if (!overlay.directionsPlan) continue;

        const key = fleetRoutePathKey(overlay.missionId);

        const path = fetched.get(key);

        if (path && path.length >= 2) {

          nextPaths.set(key, path);

          nextStates.set(key, "ready");

        } else {

          nextStates.set(key, "failed");

        }

      }

      setRoutedPathsByMissionId(nextPaths);

      setRoutedStateByMissionId(nextStates);

    })();



    return () => {

      cancelled = true;

    };

  }, [apiKey, overlays, planSignature]);



  return { routedPathsByMissionId, routedStateByMissionId };

}

