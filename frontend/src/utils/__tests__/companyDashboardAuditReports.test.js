import {
  buildAuditReport1,
  buildAuditReport3,
  ENDPOINT_BUDGET_MS,
  PAYLOAD_BUDGET_BYTES,
  REQUEST_LOAD_CLASS,
} from '../companyDashboardAuditReports';

describe('companyDashboardAuditReports', () => {
  beforeEach(() => {
    if (typeof window !== 'undefined') {
      window.__companyDashboardApiTiming = {
        requests: [
          {
            key: 'reservations',
            url: '/companies/me/reservations',
            durationMs: 1200,
            payloadBytes: 400_000,
          },
          {
            key: 'drivers',
            url: '/companies/me/drivers',
            durationMs: 250,
            payloadBytes: 80_000,
          },
        ],
      };
      window.__companyDashboardWebVitals = { lcp: 1800, inp: 120, cls: 0.02 };
    }
  });

  it('expose les budgets endpoint et classification chargement', () => {
    expect(ENDPOINT_BUDGET_MS.reservations).toBe(700);
    expect(PAYLOAD_BUDGET_BYTES).toBe(300 * 1024);
    expect(REQUEST_LOAD_CLASS.blocking).toContain('reservations');
    expect(REQUEST_LOAD_CLASS.deferred_lazy).toContain('messages');
  });

  it('marque les dépassements de budget en P0 regression candidate', () => {
    const report3 = buildAuditReport3();
    const reservations = report3.top10.find((e) => e.key === 'reservations');
    expect(reservations.status).toBe('P0 regression candidate');
    expect(report3.payloadStatus).toBe('P0 regression candidate');
  });

  it('produit le rapport 1 avec split et CWV', () => {
    const report1 = buildAuditReport1();
    expect(report1.title).toMatch(/Rapport 1/);
    expect(report1.coreWebVitals.lcp).toBe(1800);
    expect(report1.splitEstimate.apiMs).toBeGreaterThan(0);
  });
});
