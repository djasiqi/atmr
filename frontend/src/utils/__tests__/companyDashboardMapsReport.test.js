import { buildMapsReport } from '../companyDashboardMapsReport';

jest.mock('../companyDashboardPerfInstrumentation', () => ({
  isCompanyDashboardPerfEnabled: () => true,
}));

jest.mock('../companyDashboardWebPerf', () => ({
  runDashboardPerfMeasures: () => ({
    gmaps_sdk_load_ms: 1200,
    gmaps_first_markers_ms: 90,
  }),
}));

jest.mock('../companyDashboardMapsOverlayStats', () => ({
  getMapsOverlayStats: () => ({
    markerCount: 12,
    activeMarkerCount: 10,
    overlayCount: 0,
    clusterCount: null,
    markerCountAtIdleMs: 14,
  }),
}));

describe('companyDashboardMapsReport', () => {
  beforeEach(() => {
    if (typeof performance === 'undefined' || !performance.mark) return;
    ['gmaps_provider_mount', 'gmaps_sdk_request_start', 'gmaps_sdk_loaded', 'gmaps_map_loaded', 'gmaps_first_markers'].forEach(
      (name) => {
        try {
          performance.mark(name);
        } catch {
          // ignore
        }
      }
    );
  });

  it('calcule les deltas sans NaN quand marks présents', () => {
    const report = buildMapsReport();
    expect(report.markerCount).toBe(12);
    expect(report.sdkNetworkMs === null || Number.isFinite(report.sdkNetworkMs)).toBe(true);
    expect(report.interpretation).toBeInstanceOf(Array);
  });

  it('retourne null pour deltas si marks absents', () => {
    const orig = performance.getEntriesByName;
    performance.getEntriesByName = jest.fn(() => []);
    const report = buildMapsReport();
    expect(report.sdkNetworkMs).toBeNull();
    performance.getEntriesByName = orig;
  });
});
