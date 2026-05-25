/**
 * Rapport Maps ciblé — timeline SDK / carte / markers + charge overlays.
 */

import { runDashboardPerfMeasures } from './companyDashboardWebPerf';
import { getMapsOverlayStats } from './companyDashboardMapsOverlayStats';
import { isCompanyDashboardPerfEnabled } from './companyDashboardPerfInstrumentation';

function markTimeMs(name) {
  if (typeof performance === 'undefined' || !performance.getEntriesByName) return null;
  const entries = performance.getEntriesByName(name, 'mark');
  if (!entries.length) return null;
  return Math.round(entries[entries.length - 1].startTime);
}

function deltaMs(a, b) {
  if (a == null || b == null) return null;
  const d = b - a;
  return Number.isFinite(d) ? Math.round(d) : null;
}

function findMapsSdkResource() {
  if (typeof performance === 'undefined' || typeof performance.getEntriesByType !== 'function') {
    return null;
  }
  const entries = performance
    .getEntriesByType('resource')
    .filter((e) => e.name && e.name.includes('maps.googleapis.com'));
  if (!entries.length) return null;
  const last = entries[entries.length - 1];
  return {
    name: last.name,
    transferSize: last.transferSize || 0,
    durationMs: Math.round(last.duration),
  };
}

export function buildMapsReport() {
  if (!isCompanyDashboardPerfEnabled()) {
    return { enabled: false };
  }

  const measures = runDashboardPerfMeasures();
  const overlay = getMapsOverlayStats() || {};

  const providerMountAtMs = markTimeMs('gmaps_provider_mount');
  const sdkRequestStartAtMs =
    markTimeMs('gmaps_sdk_request_start') ?? markTimeMs('gmaps_sdk_start');
  const sdkLoadedAtMs = markTimeMs('gmaps_sdk_loaded') ?? markTimeMs('gmaps_sdk_end');
  const mapConstructorStartAtMs = markTimeMs('gmaps_map_constructor_start');
  const mapLoadedAtMs = markTimeMs('gmaps_map_loaded');
  const firstMarkersAtMs = markTimeMs('gmaps_first_markers');

  const sdkNetworkMs = deltaMs(sdkRequestStartAtMs, sdkLoadedAtMs);
  const mapConstructMs = deltaMs(mapConstructorStartAtMs, mapLoadedAtMs);
  const markersAfterMapMs = deltaMs(mapLoadedAtMs, firstMarkersAtMs);
  const totalMapReadyMs = deltaMs(providerMountAtMs, firstMarkersAtMs);

  return {
    title: 'Rapport Maps — timeline SDK / carte / markers',
    generatedAt: new Date().toISOString(),
    webPerfMeasures: {
      gmaps_sdk_load_ms: measures.gmaps_sdk_load_ms ?? null,
      gmaps_map_after_enable_ms: measures.gmaps_map_after_enable_ms ?? null,
      gmaps_first_markers_ms: measures.gmaps_first_markers_ms ?? null,
    },
    providerMountAtMs,
    sdkRequestStartAtMs,
    sdkLoadedAtMs,
    sdkNetworkMs,
    mapConstructorStartAtMs,
    mapLoadedAtMs,
    mapConstructMs,
    firstMarkersAtMs,
    markersAfterMapMs,
    totalMapReadyMs,
    sdkResourceEntry: findMapsSdkResource(),
    markerCount: overlay.markerCount ?? null,
    activeMarkerCount: overlay.activeMarkerCount ?? null,
    overlayCount: overlay.overlayCount ?? null,
    infoWindowCount: overlay.infoWindowCount ?? null,
    clusterCount: overlay.clusterCount ?? null,
    markerCountAtIdleMs: overlay.markerCountAtIdleMs ?? null,
    interpretation: buildInterpretation({
      sdkNetworkMs,
      mapConstructMs,
      markersAfterMapMs,
      markerCount: overlay.markerCount,
    }),
  };
}

function buildInterpretation({ sdkNetworkMs, mapConstructMs, markersAfterMapMs, markerCount }) {
  const hints = [];
  if (sdkNetworkMs != null && sdkNetworkMs >= 800) {
    hints.push('SDK réseau dominant');
  }
  if (mapConstructMs != null && mapConstructMs >= 400) {
    hints.push('construction carte lente');
  }
  if (markersAfterMapMs != null && markersAfterMapMs >= 500 && markerCount >= 50) {
    hints.push('volume markers élevé');
  }
  if (!hints.length) hints.push('pas de signal dominant — analyser les mesures brutes');
  return hints;
}

export function publishMapsReport() {
  const report = buildMapsReport();
  if (typeof window !== 'undefined') {
    window.__companyDashboardMapsReport = report;
  }
  return report;
}
