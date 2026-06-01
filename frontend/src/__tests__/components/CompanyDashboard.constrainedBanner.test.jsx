import React from 'react';
import { render, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter } from 'react-router-dom';
import { toast } from 'sonner';
import CompanyDashboard from 'pages/company/Dashboard/CompanyDashboard';
import useCompanyData from 'hooks/useCompanyData';
import useCompanyDriversForMap from 'hooks/useCompanyDriversForMap';
import useCompanySocket from 'hooks/useCompanySocket';
import useDispatchStatus from 'hooks/useDispatchStatus';
import { CONSTRAINED_IMMINENT_TOAST_ID } from 'utils/companyDriverConstrainedBanner';

global.ResizeObserver = class {
  observe() {}
  unobserve() {}
  disconnect() {}
};

jest.mock('hooks/useCompanyData');
jest.mock('hooks/useCompanyDriversForMap');
jest.mock('hooks/useCompanySocket');
jest.mock('hooks/useDispatchStatus');
jest.mock('services/companyService');
jest.mock('sonner', () => ({
  toast: {
    warning: jest.fn(),
    dismiss: jest.fn(),
    error: jest.fn(),
    success: jest.fn(),
    info: jest.fn(),
  },
}));

jest.mock('components/layout/Sidebar/CompanySidebar/CompanySidebar', () => () => <div />);
jest.mock('components/layout/Header/CompanyHeader', () => () => <div />);
jest.mock('pages/company/Dashboard/components/OverviewCards', () => () => <div data-testid="overview" />);
jest.mock('pages/company/Dashboard/components/ReservationTable', () => () => <div />);
jest.mock('pages/company/Dashboard/components/DriverLiveMap', () => () => <div />);
jest.mock('pages/driver/components/Dashboard/DriverTable', () => () => <div />);
jest.mock('components/widgets/ChatWidget', () => () => <div />);

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return ({ children }) => (
    <BrowserRouter>
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    </BrowserRouter>
  );
};

describe('CompanyDashboard bannière batterie imminente', () => {
  const mockCompany = { id: 1, name: 'Test Co' };
  const mockSocket = { on: jest.fn(), off: jest.fn(), emit: jest.fn() };

  beforeEach(() => {
    jest.clearAllMocks();
    useCompanyData.mockReturnValue({
      company: mockCompany,
      reservations: [
        {
          id: 501,
          driver_id: 99,
          status: 'assigned',
          scheduled_time: '2026-06-01T10:20:00',
        },
      ],
      driver: [],
      loadingReservations: false,
      loadingDriver: false,
      reloadReservations: jest.fn(),
      reloadDriver: jest.fn(),
      upsertReservation: jest.fn(),
    });
    useCompanyDriversForMap.mockReturnValue({
      driversForMap: [
        {
          id: 99,
          status: 'assigned_constrained',
          presence_status: 'degraded_constrained',
          latitude: 46.2,
          longitude: 6.1,
        },
      ],
      loadingDriversForMap: false,
    });
    useCompanySocket.mockReturnValue(mockSocket);
    useDispatchStatus.mockReturnValue({ label: 'Idle', progress: 0, isRunning: false });

    jest.useFakeTimers();
    jest.setSystemTime(new Date('2026-06-01T10:00:00'));
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it('affiche un toast warning sticky quand mission imminente + chauffeur constrained', async () => {
    render(<CompanyDashboard />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(toast.warning).toHaveBeenCalledWith(
        expect.stringMatching(/optimisation batterie/),
        expect.objectContaining({
          id: CONSTRAINED_IMMINENT_TOAST_ID,
          duration: Infinity,
        })
      );
    });
  });

  it('dismiss le toast quand plus aucun chauffeur constrained imminente', async () => {
    useCompanyDriversForMap.mockReturnValue({
      driversForMap: [{ id: 99, status: 'assigned', latitude: 46.2, longitude: 6.1 }],
      loadingDriversForMap: false,
    });

    render(<CompanyDashboard />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(toast.dismiss).toHaveBeenCalledWith(CONSTRAINED_IMMINENT_TOAST_ID);
    });
  });
});
