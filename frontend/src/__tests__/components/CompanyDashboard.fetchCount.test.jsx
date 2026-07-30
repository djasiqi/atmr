/**
 * Garde-fou LIRIE : budget GET au cold mount du tableau de bord entreprise.
 *
 * Périmètre (ne pas élargir sans mettre à jour les plafonds) :
 * - premier rendu de CompanyDashboard + stabilisation des queries initiales ;
 * - hors navigation interne, hors reconnexion socket, hors refetchInterval (pas de fake timers longs) ;
 * - header / sidebar mockés (stubs) pour isoler le périmètre « page dashboard » ;
 * - QueryClient de test : refetchOnWindowFocus désactivé (évite flakes CI).
 * - P5 : `useSocketConnected` mocké à false (mode dégradé) — le budget cold mount reste valide ;
 *   en prod socket connecté + entreprise, companyDrivers utilise staleTime Infinity (pas de refetch focus).
 *
 * Budget à ajuster uniquement si le contrat d’appels change volontairement (PR documentée).
 */
import React from 'react';
import { render, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import CompanyDashboard from 'pages/company/Dashboard/CompanyDashboard';
import apiClient from 'utils/apiClient';
import useCompanySocket, { useSocketConnected } from 'hooks/useCompanySocket';
import useDispatchStatus from 'hooks/useDispatchStatus';

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
jest.mock('hooks/useDispatchMode', () => ({
  useDispatchMode: () => ({ dispatchMode: 'manual' }),
}));

jest.mock('components/layout/Sidebar/CompanySidebar/CompanySidebar', () => {
  return function MockSidebar() {
    return <div data-testid="company-sidebar">Sidebar</div>;
  };
});
jest.mock('components/layout/Header/CompanyHeader', () => {
  return function MockHeader() {
    return <div data-testid="company-header">Header</div>;
  };
});
jest.mock('pages/company/Dashboard/components/OverviewCards', () => {
  return function MockOverviewCards({ stats = {} }) {
    return (
      <div data-testid="overview-cards">
        <div>Pending: {stats.pending || 0}</div>
      </div>
    );
  };
});
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
  return function MockDriverTable({ drivers }) {
    return <div data-testid="driver-table">{drivers.length} chauffeurs</div>;
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
jest.mock('pages/company/Dashboard/components/ReservationFilterBar', () => {
  return function MockReservationFilterBar() {
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
  const path = String(url || '').split('?')[0];
  return path;
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
  return ({ children }) => (
    <MemoryRouter initialEntries={['/dashboard/company/test-public-id']}>
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    </MemoryRouter>
  );
};

/** Plafonds par famille d’URL (cold mount, périmètre ci-dessus). */
export const COMPANY_DASHBOARD_GET_BUDGETS = {
  companyMe: 1,
  /** Lot 3 : bootstrap remplace le GET réservations jour. */
  companyReservations: 1,
  companyDashboardBootstrap: 1,
  companyReservationsSlash: 0,
  drivers: 0,
  driverLocations: 0,
  /** Canonique Lot 3. */
  driversLive: 1,
  /** Différé après critical-ready. */
  assignments: 1,
  delays: 1,
  realtime: 1,
  requestOffers: 1,
  /** Badge cloche limit=0 (Lot 2) — optionnel si bootstrap a pré-alimenté. */
  notificationsBadge: 1,
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

  it('respecte les plafonds GET par endpoint au cold mount', async () => {
    const buckets = createBuckets();

    const getSpy = jest.spyOn(apiClient, 'get').mockImplementation((url) => {
      tallyGet(url, buckets);
      const path = String(url || '');
      if (path.includes('/companies/me/dashboard/bootstrap')) {
        return Promise.resolve({
          data: {
            schema_version: 1,
            generated_at: '2026-01-01T00:00:00Z',
            date: '2026-01-01',
            company_id: 1,
            snapshot_cursor: 0,
            kpi: {},
            bookings: [],
            dispatch_mode: 'manual',
            notifications: { unread_count: 0 },
          },
        });
      }
      if (path.includes('/companies/me') && !path.includes('/reservations') && !path.includes('/drivers') && !path.includes('/dashboard')) {
        return Promise.resolve({ data: { id: 1, name: 'Co', public_id: 'test-public-id' } });
      }
      if (path.includes('/companies/me/reservations')) {
        return Promise.resolve({ data: [] });
      }
      if (path.includes('/companies/me/drivers/live')) {
        return Promise.resolve({ data: { drivers: [], schema_version: 1, generated_at: '', total: 0 } });
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

    render(<CompanyDashboard />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(getSpy).toHaveBeenCalled();
    });

    await waitFor(
      () => {
        expect(buckets.companyMe).toBeGreaterThanOrEqual(1);
        expect(buckets.companyDashboardBootstrap + buckets.companyReservations).toBeGreaterThanOrEqual(1);
        expect(buckets.driversLive + buckets.drivers).toBeGreaterThanOrEqual(1);
      },
      { timeout: 8000 }
    );

    // Après critical-ready, realtime/delays/offers peuvent apparaître (différés).
    // Les GET hors budget connu (ex. notifications badge) restent tolérés via `other`
    // uniquement s'ils ne sont pas des fuites massives.
    expect(buckets.other).toBeLessThanOrEqual(3);

    Object.entries(COMPANY_DASHBOARD_GET_BUDGETS).forEach(([key, max]) => {
      expect(buckets[key]).toBeLessThanOrEqual(max);
    });

    // Chemin critique total (shell inclus) : me + bootstrap + drivers/live ≤ 5
    const criticalGets =
      buckets.companyMe +
      buckets.companyDashboardBootstrap +
      buckets.driversLive +
      buckets.drivers +
      buckets.driverLocations +
      buckets.companyReservations;
    expect(criticalGets).toBeLessThanOrEqual(5);

    getSpy.mockRestore();
  });

  it('avec REACT_APP_DRIVERS_LIVE_API=1 : cold mount sur GET /drivers/live uniquement (pas drivers + locations)', async () => {
    const prevDriversLive = process.env.REACT_APP_DRIVERS_LIVE_API;
    process.env.REACT_APP_DRIVERS_LIVE_API = '1';
    try {
      const buckets = createBuckets();

      const getSpy = jest.spyOn(apiClient, 'get').mockImplementation((url) => {
        tallyGet(url, buckets);
        const path = String(url || '');
        if (path.includes('/companies/me/dashboard/bootstrap')) {
          return Promise.resolve({
            data: {
              schema_version: 1,
              generated_at: '2026-01-01T00:00:00Z',
              date: '2026-01-01',
              company_id: 1,
              snapshot_cursor: 0,
              kpi: {},
              bookings: [],
              dispatch_mode: 'manual',
              notifications: { unread_count: 0 },
            },
          });
        }
        if (path.includes('/companies/me') && !path.includes('/reservations') && !path.includes('/drivers') && !path.includes('/dashboard')) {
          return Promise.resolve({ data: { id: 1, name: 'Co', public_id: 'test-public-id' } });
        }
        if (path.includes('/companies/me/reservations')) {
          return Promise.resolve({ data: [] });
        }
        if (path.includes('/companies/me/drivers/live')) {
          return Promise.resolve({ data: { drivers: [], schema_version: 1, generated_at: '', total: 0 } });
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

      render(<CompanyDashboard />, { wrapper: createWrapper() });

      await waitFor(() => {
        expect(getSpy).toHaveBeenCalled();
      });

      await waitFor(
        () => {
          expect(buckets.companyMe).toBeGreaterThanOrEqual(1);
          expect(buckets.realtime).toBeGreaterThanOrEqual(1);
        },
        { timeout: 8000 }
      );

      expect(buckets.other).toBe(0);
      expect(buckets.driversLive).toBeGreaterThanOrEqual(1);
      expect(buckets.driversLive).toBeLessThanOrEqual(COMPANY_DASHBOARD_GET_BUDGETS.driversLive);
      expect(buckets.drivers).toBe(0);
      expect(buckets.driverLocations).toBe(0);

      Object.entries(COMPANY_DASHBOARD_GET_BUDGETS).forEach(([key, max]) => {
        expect(buckets[key]).toBeLessThanOrEqual(max);
      });

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
