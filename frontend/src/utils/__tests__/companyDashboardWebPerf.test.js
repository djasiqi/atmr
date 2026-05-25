/** @jest-environment jsdom */

describe('companyDashboardWebPerf', () => {
  beforeAll(() => {
    sessionStorage.setItem('COMPANY_DASH_PERF', '1');
    jest.resetModules();
  });

  afterAll(() => {
    sessionStorage.removeItem('COMPANY_DASH_PERF');
  });

  it('active la perf dashboard et enregistre des marks', () => {
    const { perfMark } = require('../companyDashboardWebPerf');
    const { isCompanyDashboardPerfEnabled } = require('../companyDashboardPerfInstrumentation');
    expect(isCompanyDashboardPerfEnabled()).toBe(true);
    expect(() => {
      perfMark('dashboard_start');
      perfMark('dashboard_shell_visible');
    }).not.toThrow();
    if (typeof performance !== 'undefined' && performance.getEntriesByName) {
      const marks = performance.getEntriesByName('dashboard_start', 'mark');
      expect(marks.length).toBeGreaterThanOrEqual(1);
    }
  });
});
