// frontend/tests/components/ManualBookingForm.test.jsx
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import ManualBookingForm from 'pages/company/Dashboard/components/ManualBookingForm';
import { searchClients } from 'services/companyService';
import apiClient from 'utils/apiClient';

// Mocks
jest.mock('services/companyService');
jest.mock('utils/apiClient');
jest.mock('sonner', () => ({
  toast: {
    success: jest.fn(),
    error: jest.fn(),
  },
}));

// Mock des composants complexes
jest.mock('components/common/AddressAutocomplete', () => {
  return function MockAddressAutocomplete({ value, onChange, placeholder, name }) {
    return (
      <input
        data-testid={`autocomplete-${name}`}
        value={value}
        onChange={onChange}
        placeholder={placeholder}
      />
    );
  };
});

jest.mock('components/common/EstablishmentSelect', () => {
  return function MockEstablishmentSelect({ value, onChange, placeholder }) {
    return (
      <input
        data-testid="establishment-select"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
      />
    );
  };
});

jest.mock('components/common/ServiceSelect', () => {
  return function MockServiceSelect({ placeholder }) {
    return <input data-testid="service-select" placeholder={placeholder} />;
  };
});

let lastClientSelectProps = null;
jest.mock('react-select/async-creatable', () => {
  return function MockAsyncCreatableSelect(props) {
    lastClientSelectProps = props;
    const { onChange, onCreateOption, placeholder, defaultOptions = [] } = props;
    return (
      <div data-testid="client-select">
        <select
          onChange={(e) => {
            const option = defaultOptions.find((o) => o.value === parseInt(e.target.value));
            onChange(option);
          }}
        >
          <option value="">{placeholder}</option>
          {defaultOptions.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
        <button onClick={() => onCreateOption('Nouveau Client')}>Créer</button>
      </div>
    );
  };
});

jest.mock('pages/company/Clients/components/NewClientModal', () => {
  return function MockNewClientModal({ onClose, onSave }) {
    return (
      <div data-testid="new-client-modal">
        <button onClick={() => onSave({ id: 999, first_name: 'Test', last_name: 'Client' })}>
          Enregistrer
        </button>
        <button onClick={onClose}>Fermer</button>
      </div>
    );
  };
});

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });
  return ({ children }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

describe('ManualBookingForm', () => {
  const mockClients = [
    {
      id: 1,
      first_name: 'Jean',
      last_name: 'Dupont',
      user: { first_name: 'Jean', last_name: 'Dupont' },
      billing_address: 'Rue de Lausanne 1, 1201 Genève',
    },
    {
      id: 2,
      is_institution: true,
      institution_name: 'Hôpital de la Tour',
    },
  ];

  beforeEach(() => {
    jest.clearAllMocks();
    lastClientSelectProps = null;
    searchClients.mockResolvedValue(mockClients);
    apiClient.get.mockResolvedValue({
      data: { duration: 1200, distance: 15000 }, // 20 min, 15 km
    });
  });

  it('devrait afficher le formulaire avec tous les champs', async () => {
    render(<ManualBookingForm onSuccess={jest.fn()} />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByText(/Client \*/i)).toBeInTheDocument();
    });

    expect(screen.getByText(/Lieu de prise en charge/i)).toBeInTheDocument();
    expect(screen.getByText(/Lieu de destination/i)).toBeInTheDocument();
    expect(screen.getByText(/Date & heure/i)).toBeInTheDocument();
    expect(screen.getByText(/Montant/i)).toBeInTheDocument();
    expect(screen.getByText(/Créer la réservation/i)).toBeInTheDocument();
  });

  it('devrait charger les clients par défaut', async () => {
    render(<ManualBookingForm onSuccess={jest.fn()} />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(searchClients).toHaveBeenCalledWith('');
    });

    const clientSelect = screen.getByTestId('client-select');
    expect(clientSelect).toBeInTheDocument();
  });

  it('devrait afficher une erreur si aucun client sélectionné', async () => {
    const { toast } = require('sonner');
    render(<ManualBookingForm onSuccess={jest.fn()} />, { wrapper: createWrapper() });

    const submitButton = await screen.findByText(/Créer la réservation/i);
    fireEvent.click(submitButton);

    await waitFor(() => {
      expect(toast.error).toHaveBeenCalledWith('Veuillez sélectionner un client');
    });
  });

  it('devrait activer les champs aller-retour', async () => {
    render(<ManualBookingForm onSuccess={jest.fn()} />, { wrapper: createWrapper() });

    const roundTripCheckbox = await screen.findByRole('button', { name: /Trajet AR/i });
    fireEvent.click(roundTripCheckbox);

    await waitFor(() => {
      expect(screen.getByText(/Date du retour/i)).toBeInTheDocument();
    });
  });

  it('devrait activer la récurrence', async () => {
    render(<ManualBookingForm onSuccess={jest.fn()} />, { wrapper: createWrapper() });

    const recurringCheckbox = await screen.findByRole('button', { name: /Récurrente/i });
    fireEvent.click(recurringCheckbox);

    expect(await screen.findByText(/Type de récurrence/i)).toBeInTheDocument();
    expect(screen.getByText(/Nombre de répétitions/i)).toBeInTheDocument();
  });

  it('devrait afficher la section médicale', async () => {
    render(<ManualBookingForm onSuccess={jest.fn()} />, { wrapper: createWrapper() });

    expect(await screen.findByText(/Informations médicales/i)).toBeInTheDocument();
    expect(screen.getByTestId('establishment-select')).toBeInTheDocument();
    expect(screen.getByText(/Nom du médecin/i)).toBeInTheDocument();
    expect(screen.getByText(/Notes médicales/i)).toBeInTheDocument();
  });

  it('devrait ouvrir le modal de création de client', async () => {
    render(<ManualBookingForm onSuccess={jest.fn()} />, { wrapper: createWrapper() });

    const createButton = await screen.findByText('Créer');
    fireEvent.click(createButton);

    expect(await screen.findByTestId('new-client-modal')).toBeInTheDocument();
  });

  it('devrait afficher les options chaise roulante', async () => {
    render(<ManualBookingForm onSuccess={jest.fn()} />, { wrapper: createWrapper() });

    expect(await screen.findByLabelText(/Le client est en chaise roulante/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Prendre une chaise roulante/i)).toBeInTheDocument();
  });

  /* P0.1: tests anti-régression — footer sticky + formWrapper */
  it('devrait rendre le footer sticky et le formWrapper', async () => {
    render(<ManualBookingForm onSuccess={jest.fn()} />, { wrapper: createWrapper() });

    const wrapper = await screen.findByTestId('manual-booking-form-wrapper');
    const footer = await screen.findByTestId('footer-actions-sticky');

    expect(wrapper).toBeInTheDocument();
    expect(footer).toBeInTheDocument();
  });

  it('devrait passer menuPortalTarget et styles.menuPortal au Select client', async () => {
    render(<ManualBookingForm onSuccess={jest.fn()} />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(lastClientSelectProps).not.toBeNull();
    });

    expect(lastClientSelectProps.menuPortalTarget).toBe(document.body);
    expect(lastClientSelectProps.styles).toBeDefined();
    expect(typeof lastClientSelectProps.styles.menuPortal).toBe('function');
  });

  /* P2.1: badges footer selon états */
  it('devrait afficher le badge AR quand aller-retour activé', async () => {
    render(<ManualBookingForm onSuccess={jest.fn()} />, { wrapper: createWrapper() });

    const roundTripBtn = await screen.findByRole('button', { name: /Trajet AR/i });
    fireEvent.click(roundTripBtn);

    expect(screen.getByText('AR')).toBeInTheDocument();
  });

  it('devrait afficher le badge Récurrente quand récurrence activée', async () => {
    render(<ManualBookingForm onSuccess={jest.fn()} />, { wrapper: createWrapper() });

    const recurringBtn = await screen.findByRole('button', { name: /Récurrente/i });
    fireEvent.click(recurringBtn);

    expect(screen.getByText('Récurrente')).toBeInTheDocument();
  });

  it('devrait afficher le badge Livraison quand livraison matériel activée', async () => {
    render(<ManualBookingForm onSuccess={jest.fn()} />, { wrapper: createWrapper() });

    const deliveryBtn = await screen.findByRole('button', { name: /Livraison matériel/i });
    fireEvent.click(deliveryBtn);

    expect(screen.getByText('Livraison')).toBeInTheDocument();
  });
});
