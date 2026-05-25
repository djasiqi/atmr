/**
 * Stats charge carte (markers / overlays) pour rapport Maps ciblé.
 */

let latestStats = null;

export function recordMapsOverlayStats(stats) {
  if (!stats || typeof stats !== 'object') return;
  latestStats = {
    ...stats,
    recordedAt: Date.now(),
  };
  if (typeof window !== 'undefined') {
    window.__companyDashboardMapsOverlayStats = latestStats;
  }
}

export function getMapsOverlayStats() {
  return latestStats;
}

export function resetMapsOverlayStats() {
  latestStats = null;
  if (typeof window !== 'undefined') {
    delete window.__companyDashboardMapsOverlayStats;
  }
}
