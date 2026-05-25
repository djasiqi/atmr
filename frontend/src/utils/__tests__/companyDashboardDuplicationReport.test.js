import {
  recordDashboardApiCall,
  buildDuplicationReport,
  resetDuplicationReport,
} from '../companyDashboardDuplicationReport';

jest.mock('../companyDashboardPerfInstrumentation', () => ({
  isCompanyDashboardPerfEnabled: () => true,
}));

describe('companyDashboardDuplicationReport', () => {
  beforeEach(() => {
    resetDuplicationReport();
    if (typeof performance !== 'undefined' && performance.mark) {
      try {
        performance.mark('dashboard_start');
      } catch {
        // ignore
      }
    }
  });

  it('compte deux appels même key', () => {
    recordDashboardApiCall({
      key: 'dispatch_mode',
      componentId: 'useDispatchMode',
      callerStack: 'Error\n at useDispatchMode',
    });
    recordDashboardApiCall({
      key: 'dispatch_mode',
      componentId: 'useDispatchMode',
      callerStack: 'Error\n at useDispatchMode',
    });
    const report = buildDuplicationReport();
    expect(report.perKey.dispatch_mode.count).toBe(2);
    expect(report.calls.length).toBe(2);
  });

  it('classifie strict_mode via diff dev/prod', () => {
    recordDashboardApiCall({ key: 'alerts', componentId: 'CompanyNotificationBell' });
    recordDashboardApiCall({ key: 'alerts', componentId: 'CompanyNotificationBell' });
    const report = buildDuplicationReport({
      alerts: { devCount: 2, prodCount: 1 },
      dispatch_mode: { devCount: 2, prodCount: 1 },
    });
    expect(report.perKey.alerts.cause).toBe('strict_mode');
  });
});
