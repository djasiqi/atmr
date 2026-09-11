import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import RequestDetailPanel from '../RequestDetailPanel';

const mutateBooking = jest.fn();

jest.mock('sonner', () => ({
  toast: { success: jest.fn(), error: jest.fn(), info: jest.fn() },
}));

jest.mock('../../../../components/common/AddressAutocomplete', () => {
  return function MockAddressAutocomplete({ value, onChange, placeholder, name, inputId }) {
    return (
      <input
        id={inputId}
        name={name}
        data-testid={`autocomplete-${name}`}
        value={value || ''}
        onChange={onChange}
        placeholder={placeholder}
      />
    );
  };
});

jest.mock('../../../../components/ui/InlineDatePicker', () => {
  return function MockInlineDatePicker({ value, inputId }) {
    return <input id={inputId} value={value || ''} readOnly />;
  };
});

jest.mock('../../../company/Reservations/components/BookingChat', () => {
  return function MockBookingChat() {
    return null;
  };
});

jest.mock('../../../../services/institutionSocket', () => ({
  getInstitutionSocket: () => ({ on: jest.fn(), off: jest.fn(), emit: jest.fn() }),
}));

jest.mock('../../../../utils/webAuthSession', () => ({
  getAuthEnv: () => 'app',
}));

jest.mock('../../../../hooks/useInstitutionData', () => ({
  useInstitutionRequest: jest.fn(),
  useInstitutionMe: jest.fn(),
  useSendRequest: jest.fn(),
  useCancelRequest: jest.fn(),
  useUpdateRequestBilling: jest.fn(),
  useUpdateBookingBilling: jest.fn(),
  usePatchInstitutionBooking: jest.fn(),
  useCancelInstitutionBooking: jest.fn(),
  useRequestTimeline: jest.fn(),
  useReleaseBookingForRedispatch: jest.fn(),
  useAssignExternalCarrier: jest.fn(),
  useCompleteExternalMission: jest.fn(),
  institutionQueryKeys: {
    requests: () => ['institution', 'requests'],
    requestDetail: (id) => ['institution', 'request', id],
    requestTimeline: (id) => ['institution', 'timeline', id],
  },
}));

const hooks = require('../../../../hooks/useInstitutionData');

const idleMutation = () => ({
  mutate: jest.fn(),
  mutateAsync: jest.fn(),
  isPending: false,
  isError: false,
});

const requestFixture = {
  id: 2339,
  booking_id: 45726,
  status: 'CONVERTED',
  mission_type: 'patient_transport',
  return_to_institution: true,
  pickup_location: 'Chemin des Courbes 9, 1247, Anières',
  dropoff_location: 'Hôpitaux Universitaires de Genève (HUG), Rue Gabrielle-Perret-Gentil 4',
  scheduled_time: '2026-09-12T13:15:00',
  return_time: null,
  patient: { first_name: 'Charlotte', last_name: 'CAVADINI' },
  booking_summary: {
    id: 45726,
    status: 'ACCEPTED',
    customer_name: 'Charlotte CAVADINI',
    pickup_location: 'Chemin des Courbes 9, 1247, Anières',
    dropoff_location: 'Hôpitaux Universitaires de Genève (HUG), Rue Gabrielle-Perret-Gentil 4',
    scheduled_time: '2026-09-12T13:15:00',
    time_confirmed: true,
    edit_version: 1,
    hospital_service: 'Radiologie',
    medical_facility: 'HUG',
    boarded_at: null,
  },
  legs: [
    {
      sequence_index: 0,
      pickup_location: 'Chemin des Courbes 9, 1247, Anières',
      dropoff_location: 'Hôpitaux Universitaires de Genève (HUG), Rue Gabrielle-Perret-Gentil 4',
      scheduled_time: '2026-09-12T14:00:00',
      time_confirmed: true,
      dropoff_service: 'Radiologie',
      dropoff_establishment: 'HUG',
      dropoff_doctor: '',
      destination_type: 'medical',
    },
    {
      sequence_index: 1,
      is_return_stop: true,
      pickup_location: 'Hôpitaux Universitaires de Genève (HUG), Rue Gabrielle-Perret-Gentil 4',
      dropoff_location: 'Chemin des Courbes 9, 1247, Anières',
      scheduled_time: null,
      time_confirmed: false,
    },
  ],
};

function renderPanel() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <RequestDetailPanel requestId={2339} onClose={jest.fn()} />
    </QueryClientProvider>,
  );
}

describe('RequestDetailPanel — édition RDV 14:00 → 13:00', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    hooks.useInstitutionRequest.mockReturnValue({
      data: requestFixture,
      isLoading: false,
      error: null,
    });
    hooks.useInstitutionMe.mockReturnValue({
      data: { institution_role: 'institution_admin', name: 'LHA' },
    });
    hooks.useSendRequest.mockReturnValue(idleMutation());
    hooks.useCancelRequest.mockReturnValue(idleMutation());
    hooks.useUpdateRequestBilling.mockReturnValue(idleMutation());
    hooks.useUpdateBookingBilling.mockReturnValue(idleMutation());
    hooks.useCancelInstitutionBooking.mockReturnValue(idleMutation());
    hooks.useReleaseBookingForRedispatch.mockReturnValue(idleMutation());
    hooks.useAssignExternalCarrier.mockReturnValue(idleMutation());
    hooks.useCompleteExternalMission.mockReturnValue(idleMutation());
    hooks.useRequestTimeline.mockReturnValue({ data: { events: [] }, isLoading: false });
    hooks.usePatchInstitutionBooking.mockReturnValue({
      mutate: mutateBooking,
      mutateAsync: jest.fn(),
      isPending: false,
      isError: false,
    });
  });

  it('Enregistrer envoie appointment_time 13:00, pas 14:00', async () => {
    const user = userEvent.setup();
    renderPanel();

    await user.click(screen.getByRole('button', { name: /modifier/i }));

    const destTime = document.getElementById('institution-booking-45726-dest-time-0');
    expect(destTime).toBeTruthy();
    expect(destTime.value).toBe('14:00');

    await user.clear(destTime);
    await user.type(destTime, '1300');
    expect(destTime.value).toBe('13:00');

    const saveButtons = screen.getAllByRole('button', { name: 'Enregistrer' });
    const operationalSave = saveButtons.find((btn) => !btn.disabled) || saveButtons[saveButtons.length - 1];
    await user.click(operationalSave);

    await waitFor(() => {
      expect(mutateBooking).toHaveBeenCalled();
    });

    const payload = mutateBooking.mock.calls[0][0].data;
    expect(payload.appointment_time).toBe('2026-09-12T13:00:00');
    expect(payload.leg_appointments[0].scheduled_time).toBe('2026-09-12T13:00:00');
    expect(payload.appointment_time).not.toBe('2026-09-12T14:00:00');
    expect(payload.leg_appointments[0].scheduled_time).not.toBeUndefined();
    expect(payload.leg_appointments[0].scheduled_time).not.toBeNull();
    expect(mutateBooking.mock.calls[0][0].bookingId).toBe(45726);
  });
});
