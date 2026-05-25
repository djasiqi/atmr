/**
 * Bootstrap perf dashboard — augmente le buffer Resource Timing avant collecte bundle.
 */

import { isCompanyDashboardPerfEnabled } from './companyDashboardPerfInstrumentation';

const DEFAULT_BUFFER_SIZE = 3000;
let bootstrapped = false;
let resourceObserver = null;
const streamedResources = [];

export function getStreamedResourceEntries() {
  return streamedResources.slice();
}

export function getResourceBufferDiagnostics() {
  if (typeof performance === 'undefined') {
    return {
      initialBufferSize: null,
      observedEntries: 0,
      droppedDuringObservation: 0,
      bufferFullEvents: 0,
    };
  }
  const entries = performance.getEntriesByType('resource');
  const observedEntries = entries.length + streamedResources.length;
  const bufferSize =
    typeof performance.resourceTimingBufferSize === 'number'
      ? performance.resourceTimingBufferSize
      : null;
  const store = typeof window !== 'undefined' ? window.__companyDashboardResourceBuffer : null;
  return {
    initialBufferSize: store?.initialBufferSize ?? bufferSize,
    observedEntries,
    droppedDuringObservation: store?.droppedDuringObservation ?? 0,
    bufferFullEvents: store?.bufferFullEvents ?? 0,
  };
}

/**
 * À appeler dès l'activation perf (montage dashboard).
 */
export function bootstrapCompanyDashboardPerf() {
  if (bootstrapped || typeof performance === 'undefined') return;
  if (!isCompanyDashboardPerfEnabled()) return;
  bootstrapped = true;

  const initialBufferSize =
    typeof performance.resourceTimingBufferSize === 'number'
      ? performance.resourceTimingBufferSize
      : 250;

  if (typeof performance.setResourceTimingBufferSize === 'function') {
    try {
      performance.setResourceTimingBufferSize(DEFAULT_BUFFER_SIZE);
    } catch {
      // ignore unsupported
    }
  }

  if (typeof window !== 'undefined') {
    window.__companyDashboardResourceBuffer = {
      initialBufferSize,
      droppedDuringObservation: 0,
      bufferFullEvents: 0,
    };
  }

  if (typeof PerformanceObserver !== 'undefined') {
    try {
      resourceObserver = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          streamedResources.push(entry);
        }
        const buf = typeof window !== 'undefined' ? window.__companyDashboardResourceBuffer : null;
        if (buf && list.getEntries().length > 0) {
          const resources = performance.getEntriesByType('resource');
          if (resources.length >= DEFAULT_BUFFER_SIZE - 10) {
            buf.bufferFullEvents += 1;
          }
        }
      });
      resourceObserver.observe({ type: 'resource', buffered: true });
    } catch {
      resourceObserver = null;
    }
  }
}

export function teardownCompanyDashboardPerfBootstrap() {
  if (resourceObserver) {
    try {
      resourceObserver.disconnect();
    } catch {
      // ignore
    }
    resourceObserver = null;
  }
}
