import { trackClientKpiEvent } from './clientKpi';

describe('clientKpi', () => {
  beforeEach(() => {
    window.__LIRIE_CLIENT_KPI__ = [];
  });

  it('émet un événement structuré', () => {
    const event = trackClientKpiEvent('history_export_clicked', { period: 'this_month' });
    expect(event.name).toBe('history_export_clicked');
    expect(event.period).toBe('this_month');
    expect(typeof event.at).toBe('number');
    expect(window.__LIRIE_CLIENT_KPI__).toHaveLength(1);
    expect(window.__LIRIE_CLIENT_KPI__[0].name).toBe('history_export_clicked');
  });
});

