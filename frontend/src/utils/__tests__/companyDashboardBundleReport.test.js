import { buildBundleReport } from '../companyDashboardBundleReport';

jest.mock('../companyDashboardPerfInstrumentation', () => ({
  isCompanyDashboardPerfEnabled: () => true,
}));

jest.mock('../companyDashboardPerfBootstrap', () => ({
  getResourceBufferDiagnostics: () => ({
    initialBufferSize: 3000,
    observedEntries: 2,
    droppedDuringObservation: 0,
    bufferFullEvents: 0,
  }),
  getStreamedResourceEntries: () => [],
}));

describe('companyDashboardBundleReport', () => {
  beforeEach(() => {
    if (typeof performance !== 'undefined' && performance.clearResourceTimings) {
      performance.clearResourceTimings();
    }
  });

  it('classifie prefetch vs critical', () => {
    const entries = [
      {
        name: 'http://localhost/static/js/main.app.js',
        initiatorType: 'script',
        transferSize: 1000,
        encodedBodySize: 900,
        decodedBodySize: 900,
        startTime: 100,
        responseEnd: 200,
        duration: 100,
        nextHopProtocol: 'h2',
      },
      {
        name: 'http://localhost/static/js/CompanyInvoices.chunk.js',
        initiatorType: 'link',
        transferSize: 256000,
        encodedBodySize: 250000,
        decodedBodySize: 250000,
        startTime: 150,
        responseEnd: 400,
        duration: 250,
        nextHopProtocol: 'h2',
      },
    ];
    performance.getEntriesByType = jest.fn((type) =>
      type === 'resource' ? entries : []
    );
    performance.getEntriesByName = jest.fn(() => [{ startTime: 50 }]);

    const report = buildBundleReport();
    expect(report.bytesPrefetched).toBeGreaterThan(0);
    expect(report.top20.length).toBeGreaterThan(0);
    expect(report.bufferDiagnostics.droppedDuringObservation).toBe(0);
  });
});
