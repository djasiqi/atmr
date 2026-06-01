import {
  countConstrainedAssignedImminentDrivers,
  buildConstrainedImminentToastMessage,
  CONSTRAINED_IMMINENT_WINDOW_MS,
} from '../companyDriverConstrainedBanner';

describe('companyDriverConstrainedBanner', () => {
  const nowMs = Date.parse('2026-06-01T10:00:00');

  it('compte un chauffeur assigned_constrained avec mission dans < 30 min', () => {
    const drivers = [
      {
        id: 42,
        status: 'assigned_constrained',
        presence_status: 'degraded_constrained',
      },
    ];
    const reservations = [
      {
        id: 100,
        driver_id: 42,
        status: 'assigned',
        scheduled_time: '2026-06-01T10:20:00',
      },
    ];
    expect(
      countConstrainedAssignedImminentDrivers(drivers, reservations, nowMs)
    ).toBe(1);
  });

  it('ignore les chauffeurs constrained sans mission imminente', () => {
    const drivers = [
      {
        id: 42,
        status: 'assigned_constrained',
        presence_status: 'degraded_constrained',
      },
    ];
    const reservations = [
      {
        id: 100,
        driver_id: 42,
        scheduled_time: '2026-06-01T12:00:00',
      },
    ];
    expect(
      countConstrainedAssignedImminentDrivers(drivers, reservations, nowMs)
    ).toBe(0);
  });

  it('ignore les chauffeurs non assignés même si constrained', () => {
    const drivers = [
      {
        id: 7,
        status: 'available_constrained',
        presence_status: 'degraded_constrained',
      },
    ];
    const reservations = [
      {
        id: 101,
        driver_id: 7,
        scheduled_time: '2026-06-01T10:15:00',
      },
    ];
    expect(
      countConstrainedAssignedImminentDrivers(drivers, reservations, nowMs)
    ).toBe(0);
  });

  it('buildConstrainedImminentToastMessage pluralise correctement', () => {
    expect(buildConstrainedImminentToastMessage(0)).toBe('');
    expect(buildConstrainedImminentToastMessage(1)).toMatch(/1 chauffeur ASSIGNED a/);
    expect(buildConstrainedImminentToastMessage(2)).toMatch(/2 chauffeurs ASSIGNED ont/);
    expect(buildConstrainedImminentToastMessage(1)).toMatch(/optimisation batterie/);
  });

  it('respecte la fenêtre configurable', () => {
    const drivers = [
      {
        id: 1,
        status: 'assigned',
        presence_status: 'degraded_constrained',
      },
    ];
    const reservations = [
      {
        driver_id: 1,
        scheduled_time: '2026-06-01T10:25:00',
      },
    ];
    expect(
      countConstrainedAssignedImminentDrivers(
        drivers,
        reservations,
        nowMs,
        CONSTRAINED_IMMINENT_WINDOW_MS
      )
    ).toBe(1);
    expect(
      countConstrainedAssignedImminentDrivers(drivers, reservations, nowMs, 10 * 60 * 1000)
    ).toBe(0);
  });
});
