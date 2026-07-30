/**
 * Garde-fou LIRIE : budget GET au cold mount du tableau de bord entreprise.
 *
 * Périmètre :
 * - premier rendu de CompanyDashboard (sans mock shell Header/Sidebar/DispatchMode) ;
 * - widgets lourds mockés (carte, tables, graphiques…) ;
 * - QueryClient de test : refetchOnWindowFocus désactivé ;
 * - `useSocketConnected` mocké à false (mode dégradé).
 *
 * Deux familles :
 * - **critiques** : avant `dashboard-critical-ready` (me, bootstrap v2, drivers/live) ≤ 5 GET ;
 * - **différées** : delays / offers / realtime / assignments uniquement après critical-ready
 *   (`deferredQueriesEnabled`).
 */
import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import CompanyDashboard from 'pages/company/Dashboard/CompanyDashboard';
import apiClient from 'utils/apiClient';
import useCompanySocket, { useSocketConnected } from 'hooks/useCompanySocket';
import useDispatchStatus from 'hooks/useDispatchStatus';
import { lirieKeys } from '../../queryKeys/lirie';

jest.mock('hooks/useCompanySocket', () => ({
  __esModule: true,
  default: jest.fn(() => ({ on: jest.fn(), off: jest.fn(), emit: jest.fn() })),
  useSocketConnected: jest.fn(() => false),
}));
jest.mock('hooks/useDispatchStatus');
jest.mock('hooks/useCompanyAuthToken', () => ({
  __esModule: true,
  default: () => ({
    user: { id: 1, isCompany: true, role: 'company' },
    isCompanyAuthReady: true,
  }),
}));

jest.mock('pages/company/Dashboard/components/ReservationTable', () => {
  return function MockReservationTable({ reservations }) {
    return <div data-testid="reservation-table">{reservations.length} réservations</div>;
  };
});
jest.mock('pages/company/Dashboard/components/DriverLiveMap', () => {
  return function MockDriverLiveMap() {
    return <div data-testid="driver-live-map">Carte</div>;
  };
});
jest.mock('pages/driver/components/Dashboard/DriverTable', () => {
  return function MockDriverTable({ driver }) {
    return <div data-testid="driver-table">{driver?.length ?? 0} chauffeurs</div>;
  };
});
jest.mock('pages/company/Dashboard/components/ManualBookingForm', () => {
  return function MockManualBookingForm() {
    return <div data-testid="manual-booking-form">Form</div>;
  };
});
jest.mock('components/widgets/ChatWidget', () => {
  return function MockChatWidget() {
    return <div data-testid="chat-widget">Chat</div>;
  };
});
jest.mock('pages/company/Dashboard/components/OpportunitiesSection', () => {
  return function MockOpportunitiesSection() {
    return null;
  };
});
jest.mock('pages/company/Dashboard/components/InstitutionOffersTable', () => {
  return function MockInstitutionOffersTable() {
    return null;
  };
});
jest.mock('pages/company/Dashboard/components/QuickAssignPanel', () => {
  return function MockQuickAssignPanel() {
    return null;
  };
});
jest.mock('pages/company/Dashboard/components/DispatchModeStatusBar', () => {
  return function MockDispatchModeStatusBar() {
    return null;
  };
});
jest.mock('components/demo/DemoInteractiveGuide', () => {
  return function MockDemo() {
    return null;
  };
});
jest.mock('pages/company/Dashboard/components/ReservationChart', () => {
  return function MockReservationChart() {
    return null;
  };
});
jest.mock('components/reservations/ReservationModals', () => {
  return function MockReservationModals() {
    return null;
  };
});
jest.mock('components/reservations/TransferBookingModal', () => {
  return function MockTransferBookingModal() {
    return null;
  };
});
jest.mock('components/ui/InlineDatePicker', () => {
  return function MockInlineDatePicker({ value, onChange }) {
    return (
      <button type="button" data-testid="inline-date" onClick={() => onChange?.(value)}>
        {value}
      </button>
    );
  };
});

global.ResizeObserver = class {
  observe() {}
  unobserve() {}
  disconnect() {}
};

function fakeCompanyJwt() {
  const payload = { exp: Math.floor(Date.now() / 1000) + 999999, role: 'company', sub: '1' };
  const header = btoa(JSON.stringify({ alg: 'HS256', typ: 'JWT' }));
  const body = btoa(JSON.stringify(payload));
  return `${header}.${body}.sig`;
}

function normalizeGetUrl(url) {
  return String(url || '').split('?')[0];
}

function createBuckets() {
  return {
    companyMe: 0,
    companyReservations: 0,
    companyReservationsSlash: 0,
    companyDashboardBootstrap: 0,
    drivers: 0,
    driverLocations: 0,
    driversLive: 0,
    assignments: 0,
    delays: 0,
    realtime: 0,
    requestOffers: 0,
    notificationsBadge: 0,
    other: 0,
  };
}

function tallyGet(url, buckets) {
  const p = normalizeGetUrl(url);
  if (p === '/companies/me' || p.endsWith('/companies/me')) buckets.companyMe += 1;
  else if (p.includes('/companies/me/dashboard/bootstrap')) buckets.companyDashboardBootstrap += 1;
  else if (p.includes('/companies/me/reservations') && p.includes('/reservations/') && !p.includes('summary'))
    buckets.companyReservationsSlash += 1;
  else if (p.includes('/companies/me/reservations') && !p.includes('summary')) buckets.companyReservations += 1;
  else if (p.includes('/companies/me/drivers/live')) buckets.driversLive += 1;
  else if (p.includes('/companies/me/drivers/locations')) buckets.driverLocations += 1;
  else if (p.includes('/companies/me/drivers')) buckets.drivers += 1;
  else if (p.includes('/company_dispatch/assignments')) buckets.assignments += 1;
  else if (p.includes('/company_dispatch/delays') && !p.includes('/live')) buckets.delays += 1;
  else if (p.includes('/company_dispatch/dashboard/realtime')) buckets.realtime += 1;
  else if (p.includes('/company/request-offers') || p.includes('/request-offers')) buckets.requestOffers += 1;
  else if (p.includes('/companies/notifications')) buckets.notificationsBadge += 1;
  else buckets.other += 1;
}

const DEFERRED_BUCKET_KEYS = ['assignments', 'delays', 'realtime', 'requestOffers'];

/** GET critiques avant `dashboard-critical-ready`. */
export const COMPANY_DASHBOARD_CRITICAL_GET_BUDGETS = {
  companyMe: 1,
  /** StrictMode (dev/test) peut remonter le composant une 2ᵉ fois. */
  companyDashboardBootstrap: 2,
  companyReservations: 0,
  companyReservationsSlash: 0,
  drivers: 0,
  driverLocations: 0,
  driversLive: 1,
  notificationsBadge: 0,
};

/** GET différées après critical-ready (`deferredQueriesEnabled`). */
export const COMPANY_DASHBOARD_DEFERRED_GET_BUDGETS = {
  /** fetchAssignedReservations — peut remonter à 2 sous StrictMode (dev). */
  assignments: 2,
  delays: 1,
  realtime: 1,
  requestOffers: 1,
  /** Fenêtre pending multi-jours (onglet À décider) + ensureQueryData assignés. */
  companyReservations: 2,
};

/** Union rétrocompatible pour les plafonds globaux post-mount. */
export const COMPANY_DASHBOARD_GET_BUDGETS = {
  ...COMPANY_DASHBOARD_CRITICAL_GET_BUDGETS,
  ...COMPANY_DASHBOARD_DEFERRED_GET_BUDGETS,
  notificationsBadge: 1,
};

function bootstrapV2Payload(overrides = {}) {
  return {
    schema_version: 2,
    generated_at: '2026-01-01T00:00:00Z',
    date: '2026-01-01',
    company_id: 1,
    snapshot_cursor: 0,
    kpi: {
      pending_decision: 0,
      unassigned: 0,
      delay_count: 0,
      critical_delay_count: 0,
      critical_delay_minutes: 15,
      in_service: 0,
    },
    summary: { to_handle: 0 },
    action_queue: [],
    action_queue_total: 0,
    action_queue_truncated: false,
    bookings: [],
    bookings_truncated: false,
    bookings_limit: 500,
    bookings_returned: 0,
    bookings_total: 0,
    dispatch_mode: 'manual',
    notifications: { unread_count: 0 },
    health: { realtime_sequence: 'ok' },
    ...overrides,
  };
}

function isCriticalGetUrl(url) {
  const p = normalizeGetUrl(url);
  if (p === '/companies/me' || p.endsWith('/companies/me')) return true;
  if (p.includes('/companies/me/dashboard/bootstrap')) return true;
  if (p.includes('/companies/me/drivers/live')) return true;
  if (p.includes('/companies/me/drivers') && !p.includes('locations') && !p.includes('live')) {
    return true;
  }
  return false;
}

function createApiGetMock(callLog) {
  return jest.spyOn(apiClient, 'get').mockImplementation((url) => {
    callLog.push(String(url || ''));

    const path = String(url || '');
    if (path.includes('/companies/me/dashboard/bootstrap')) {
      return Promise.resolve({ data: bootstrapV2Payload() });
    }
    if (
      path.includes('/companies/me') &&
      !path.includes('/reservations') &&
      !path.includes('/drivers') &&
      !path.includes('/dashboard')
    ) {
      return Promise.resolve({ data: { id: 1, name: 'Co', public_id: 'test-public-id' } });
    }
    if (path.includes('/companies/me/reservations')) {
      return Promise.resolve({ data: [] });
    }
    if (path.includes('/companies/me/drivers/live')) {
      return Promise.resolve({
        data: { drivers: [], schema_version: 1, generated_at: '', total: 0 },
      });
    }
    if (path.includes('/companies/me/drivers/locations')) {
      return Promise.resolve({ data: [] });
    }
    if (path.includes('/companies/me/drivers') && !path.includes('locations') && !path.includes('live')) {
      return Promise.resolve({ data: [] });
    }
    if (path.includes('/company_dispatch/assignments')) {
      return Promise.resolve({ data: [] });
    }
    if (path.includes('/company_dispatch/delays') && !path.includes('/live')) {
      return Promise.resolve({ data: [] });
    }
    if (path.includes('/company_dispatch/dashboard/realtime')) {
      return Promise.resolve({
        data: {
          quality_metrics: null,
          opportunities: [],
          current_delays: [],
          driver_load: [],
          stats: null,
        },
      });
    }
    if (path.includes('/company/request-offers') || path.includes('request-offers')) {
      return Promise.resolve({ data: { offers: [], total: 0 } });
    }
    return Promise.resolve({ data: null });
  });
}

function bucketsFromCallLog(callLog) {
  const buckets = createBuckets();
  callLog.forEach((url) => tallyGet(url, buckets));
  return buckets;
}

function splitCriticalAndDeferredCalls(callLog) {
  const firstDeferredIdx = callLog.findIndex((url) => !isCriticalGetUrl(url));
  const criticalCalls =
    firstDeferredIdx === -1 ? callLog : callLog.slice(0, firstDeferredIdx);
  const deferredCalls =
    firstDeferredIdx === -1 ? [] : callLog.slice(firstDeferredIdx);
  return { criticalCalls, deferredCalls };
}

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        refetchOnWindowFocus: false,
      },
      mutations: { retry: false },
    },
  });
  // Bootstrap alimente ce cache — évite GET /company_dispatch/mode au montage (useDispatchMode réel).
  queryClient.setQueryData(lirieKeys.dispatchMode(), 'manual');
  return ({ children }) => (
    <MemoryRouter initialEntries={['/dashboard/company/test-public-id']}>
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    </MemoryRouter>
  );
};

describe('CompanyDashboard — budget GET (apiClient)', () => {
  const mockSocket = { on: jest.fn(), off: jest.fn(), emit: jest.fn() };

  beforeEach(() => {
    jest.clearAllMocks();
    localStorage.setItem('user', JSON.stringify({ id: 1, role: 'company' }));
    localStorage.setItem('company_access_token', fakeCompanyJwt());
    useCompanySocket.mockReturnValue(mockSocket);
    useSocketConnected.mockReturnValue(false);
    useDispatchStatus.mockReturnValue({ label: 'Idle', progress: 0, isRunning: false });
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('respecte le budget GET critique avant dashboard-critical-ready et diffère le reste', async () => {
    const callLog = [];
    const getSpy = createApiGetMock(callLog);

    render(<CompanyDashboard />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByTestId('dashboard-critical-ready')).toBeInTheDocument();
    });

    await waitFor(
      () => {
        const interim = bucketsFromCallLog(callLog);
        expect(interim.delays).toBeGreaterThanOrEqual(1);
        expect(interim.realtime).toBeGreaterThanOrEqual(1);
        expect(interim.requestOffers).toBeGreaterThanOrEqual(1);
      },
      { timeout: 8000 }
    );

    const { criticalCalls, deferredCalls } = splitCriticalAndDeferredCalls(callLog);
    const bucketsBefore = bucketsFromCallLog(criticalCalls);
    const bucketsAfter = bucketsFromCallLog(callLog);

    Object.entries(COMPANY_DASHBOARD_CRITICAL_GET_BUDGETS).forEach(([key, max]) => {
      expect(bucketsBefore[key]).toBeLessThanOrEqual(max);
    });

    expect(deferredCalls.length).toBeGreaterThan(0);
    DEFERRED_BUCKET_KEYS.forEach((key) => {
      expect(bucketsBefore[key]).toBe(0);
    });

    const criticalGetsBeforeReady =
      bucketsBefore.companyMe +
      bucketsBefore.companyDashboardBootstrap +
      bucketsBefore.driversLive +
      bucketsBefore.drivers +
      bucketsBefore.driverLocations +
      bucketsBefore.companyReservations;
    expect(criticalGetsBeforeReady).toBeLessThanOrEqual(5);
    expect(criticalGetsBeforeReady).toBeGreaterThanOrEqual(3);

    Object.entries(COMPANY_DASHBOARD_GET_BUDGETS).forEach(([key, max]) => {
      expect(bucketsAfter[key]).toBeLessThanOrEqual(max);
    });

    expect(bucketsAfter.delays).toBeGreaterThanOrEqual(1);
    expect(bucketsAfter.realtime).toBeGreaterThanOrEqual(1);
    expect(bucketsAfter.requestOffers).toBeGreaterThanOrEqual(1);
    expect(bucketsAfter.other).toBeLessThanOrEqual(3);

    const bootstrapCall = getSpy.mock.calls.find((c) =>
      String(c[0] || '').includes('/companies/me/dashboard/bootstrap')
    );
    expect(bootstrapCall?.[1]?.params?.schema_version).toBe(2);

    getSpy.mockRestore();
  });

  it('avec REACT_APP_DRIVERS_LIVE_API=1 : cold mount sur GET /drivers/live uniquement', async () => {
    const prevDriversLive = process.env.REACT_APP_DRIVERS_LIVE_API;
    process.env.REACT_APP_DRIVERS_LIVE_API = '1';
    try {
      const callLog = [];
      const getSpy = createApiGetMock(callLog);

      render(<CompanyDashboard />, { wrapper: createWrapper() });

      await waitFor(() => {
        expect(screen.getByTestId('dashboard-critical-ready')).toBeInTheDocument();
      });

      await waitFor(
        () => {
          expect(splitCriticalAndDeferredCalls(callLog).deferredCalls.length).toBeGreaterThan(0);
        },
        { timeout: 8000 }
      );

      const { criticalCalls, deferredCalls } = splitCriticalAndDeferredCalls(callLog);
      const bucketsBefore = bucketsFromCallLog(criticalCalls);
      const bucketsAfter = bucketsFromCallLog(callLog);

      expect(deferredCalls.length).toBeGreaterThan(0);
      expect(bucketsBefore.driversLive).toBeGreaterThanOrEqual(1);
      expect(bucketsBefore.drivers).toBe(0);
      expect(bucketsBefore.driverLocations).toBe(0);
      expect(bucketsAfter.driversLive).toBeLessThanOrEqual(COMPANY_DASHBOARD_GET_BUDGETS.driversLive);
      expect(bucketsAfter.drivers).toBe(0);
      expect(bucketsAfter.driverLocations).toBe(0);

      getSpy.mockRestore();
    } finally {
      if (prevDriversLive === undefined) {
        delete process.env.REACT_APP_DRIVERS_LIVE_API;
      } else {
        process.env.REACT_APP_DRIVERS_LIVE_API = prevDriversLive;
      }
    }
  });
});
