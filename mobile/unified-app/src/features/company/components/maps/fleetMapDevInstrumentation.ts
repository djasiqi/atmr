/** Compteurs dev pour mesurer rerenders / clustering (Sprint perf carte flotte). */

let driverMarkerRenderCount = 0;
let clusterFleetMarkersCount = 0;

export function countDriverMarkerRender(driverId: number): void {
  if (typeof __DEV__ === "undefined" || !__DEV__) return;
  driverMarkerRenderCount += 1;
  // eslint-disable-next-line no-console -- instrumentation dev volontaire
  console.count(`[fleet-map] DriverMarker:${driverId}`);
}

export function countClusterFleetMarkers(): void {
  if (typeof __DEV__ === "undefined" || !__DEV__) return;
  clusterFleetMarkersCount += 1;
  // eslint-disable-next-line no-console -- instrumentation dev volontaire
  console.count("[fleet-map] clusterFleetMarkers");
}

export function resetFleetMapDevCounters(): void {
  driverMarkerRenderCount = 0;
  clusterFleetMarkersCount = 0;
}

export function getFleetMapDevCounters(): { driverMarkerRenders: number; clusterRuns: number } {
  return {
    driverMarkerRenders: driverMarkerRenderCount,
    clusterRuns: clusterFleetMarkersCount,
  };
}
