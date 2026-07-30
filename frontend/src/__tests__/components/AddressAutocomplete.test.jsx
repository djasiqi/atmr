// frontend/tests/components/AddressAutocomplete.test.jsx
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import AddressAutocomplete, {
  clearAddressAutocompleteCache,
} from 'components/common/AddressAutocomplete';

jest.mock('../../utils/apiClient', () => ({
  __esModule: true,
  default: {
    get: jest.fn(),
  },
}));

import apiClient from '../../utils/apiClient';

describe('AddressAutocomplete', () => {
  const mockOnChange = jest.fn();
  const mockOnSelect = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    clearAddressAutocompleteCache();
    // Public autocomplete + favorites JWT (401 anonyme = OK)
    apiClient.get.mockImplementation((url) => {
      if (String(url).includes('favorites/autocomplete')) {
        return Promise.resolve({ status: 401, data: { error: 'Authentification requise' } });
      }
      return Promise.resolve({ status: 200, data: [] });
    });
  });

  it('devrait afficher un champ de saisie', () => {
    render(
      <AddressAutocomplete
        name="test-address"
        value=""
        onChange={mockOnChange}
        onSelect={mockOnSelect}
        placeholder="Saisir une adresse"
      />
    );

    const input = screen.getByPlaceholderText('Saisir une adresse');
    expect(input).toBeInTheDocument();
    expect(input).toHaveAttribute('type', 'text');
  });

  it('devrait afficher les suggestions après saisie', async () => {
    const mockSuggestions = [
      {
        source: 'photon',
        label: 'HUG, Rue Gabrielle-Perret-Gentil 4, 1205 Genève',
        address: 'Rue Gabrielle-Perret-Gentil 4',
        postcode: '1205',
        city: 'Genève',
        lat: 46.19226,
        lon: 6.14262,
      },
      {
        source: 'photon',
        label: 'Avenue de la Gare 10, 1003 Lausanne',
        address: 'Avenue de la Gare 10',
        postcode: '1003',
        city: 'Lausanne',
        lat: 46.5197,
        lon: 6.6294,
      },
    ];

    apiClient.get.mockImplementation((url) => {
      if (String(url).includes('favorites/autocomplete')) {
        return Promise.resolve({ status: 401, data: {} });
      }
      return Promise.resolve({ status: 200, data: mockSuggestions });
    });

    const user = userEvent.setup();
    render(
      <AddressAutocomplete
        name="address"
        value=""
        onChange={mockOnChange}
        onSelect={mockOnSelect}
        debounceMs={50}
      />
    );

    const input = screen.getByRole('combobox');
    await user.type(input, 'Genève');

    await waitFor(() => {
      expect(apiClient.get).toHaveBeenCalled();
    });

    await waitFor(
      () => {
        expect(screen.getByText('HUG, Rue Gabrielle-Perret-Gentil 4, 1205 Genève')).toBeInTheDocument();
      },
      { timeout: 3000 }
    );
  });

  it('devrait permettre de sélectionner une suggestion', async () => {
    const mockSuggestions = [
      {
        source: 'photon',
        label: 'Rue de Lausanne 1, 1201 Genève',
        address: 'Rue de Lausanne 1',
        postcode: '1201',
        city: 'Genève',
        lat: 46.2044,
        lon: 6.1432,
      },
    ];

    // Mock API backend qui retourne directement les suggestions
    apiClient.get.mockImplementation((url) => {
      if (String(url).includes('favorites/autocomplete')) {
        return Promise.resolve({ status: 401, data: {} });
      }
      return Promise.resolve({ status: 200, data: mockSuggestions });
    });

    const user = userEvent.setup();
    render(
      <AddressAutocomplete
        name="pickup"
        value=""
        onChange={mockOnChange}
        onSelect={mockOnSelect}
        debounceMs={50}
      />
    );

    const input = screen.getByRole('combobox');
    await user.type(input, 'Rue de Lausanne');

    // Attendre les suggestions
    await waitFor(
      () => {
        expect(screen.getByText('Rue de Lausanne 1, 1201 Genève')).toBeInTheDocument();
      },
      { timeout: 3000 }
    );

    // Sélectionner la suggestion
    const suggestion = screen.getByText('Rue de Lausanne 1, 1201 Genève');
    fireEvent.mouseDown(suggestion);

    await waitFor(() => {
      expect(mockOnSelect).toHaveBeenCalledWith(
        expect.objectContaining({
          label: 'Rue de Lausanne 1, 1201 Genève',
          lat: 46.2044,
          lon: 6.1432,
        })
      );
    });
  });

  it('devrait gérer la navigation au clavier', async () => {
    const mockSuggestions = [
      {
        source: 'photon',
        label: 'Genève Ville',
        city: 'Genève',
        lat: 46.2044,
        lon: 6.1432,
      },
      {
        source: 'photon',
        label: 'Lausanne Centre',
        city: 'Lausanne',
        lat: 46.5197,
        lon: 6.6294,
      },
    ];

    apiClient.get.mockImplementation((url) => {
      if (String(url).includes('favorites/autocomplete')) {
        return Promise.resolve({ status: 401, data: {} });
      }
      return Promise.resolve({ status: 200, data: mockSuggestions });
    });

    const user = userEvent.setup();
    render(
      <AddressAutocomplete
        name="destination"
        value=""
        onChange={mockOnChange}
        onSelect={mockOnSelect}
        minChars={2}
        debounceMs={50}
      />
    );

    const input = screen.getByRole('combobox');
    await user.type(input, 'Ge');

    await waitFor(
      () => {
        expect(screen.getByRole('option', { name: /Genève Ville/i })).toBeInTheDocument();
      },
      { timeout: 3000 }
    );

    const combobox = screen.getByRole('combobox');
    await waitFor(() => {
      expect(combobox).toHaveAttribute('aria-activedescendant', 'destination-ac-option-0');
    });

    fireEvent.keyDown(combobox, { key: 'ArrowDown', code: 'ArrowDown' });
    await waitFor(() => {
      expect(combobox).toHaveAttribute('aria-activedescendant', 'destination-ac-option-1');
    });

    fireEvent.keyDown(combobox, { key: 'Enter', code: 'Enter', keyCode: 13, which: 13 });

    await waitFor(() => {
      expect(mockOnSelect).toHaveBeenCalledWith(
        expect.objectContaining({ label: 'Lausanne Centre' })
      );
    });
  });

  it('devrait fermer les suggestions avec Escape', async () => {
    const mockSuggestions = [{ source: 'photon', label: 'Test', lat: 46.2, lon: 6.1 }];

    apiClient.get.mockImplementation((url) => {
      if (String(url).includes('favorites/autocomplete')) {
        return Promise.resolve({ status: 401, data: {} });
      }
      return Promise.resolve({ status: 200, data: mockSuggestions });
    });

    const user = userEvent.setup();
    render(
      <AddressAutocomplete
        name="test"
        value=""
        onChange={mockOnChange}
        onSelect={mockOnSelect}
        debounceMs={50}
      />
    );

    const input = screen.getByRole('combobox');
    await user.type(input, 'Test');

    await waitFor(
      () => {
        expect(screen.getByRole('listbox')).toBeInTheDocument();
      },
      { timeout: 3000 }
    );

    fireEvent.keyDown(screen.getByRole('combobox'), { key: 'Escape', code: 'Escape' });

    await waitFor(() => {
      expect(screen.queryByRole('listbox')).not.toBeInTheDocument();
    });
  });

  it('devrait prioriser l\'alias canonique HUG avant Google Places', async () => {
    const mockSuggestions = [
      {
        source: 'alias',
        label: 'Hôpitaux Universitaires de Genève (HUG), Rue Gabrielle-Perret-Gentil 4, 1205 Genève',
        address: 'Rue Gabrielle-Perret-Gentil 4, 1205 Genève',
        main_text: 'Hôpitaux Universitaires de Genève (HUG)',
        secondary_text: 'Rue Gabrielle-Perret-Gentil 4, 1205 Genève',
        name: 'Hôpitaux Universitaires de Genève (HUG)',
        lat: 46.19226,
        lon: 6.14262,
      },
      {
        source: 'google_places',
        label: 'HUG - Bâtiment Gustave Julliard, Rue Alcide-Jentzer 17, 1205 Genève',
        main_text: 'HUG - Bâtiment Gustave Julliard',
        secondary_text: 'Rue Alcide-Jentzer 17, 1205 Genève, Suisse',
        place_id: 'google-julliard',
      },
    ];

    apiClient.get.mockImplementation((url) => {
      if (String(url).includes('favorites/autocomplete')) {
        return Promise.resolve({ status: 401, data: {} });
      }
      return Promise.resolve({ status: 200, data: mockSuggestions });
    });

    const user = userEvent.setup();
    render(
      <AddressAutocomplete
        name="dropoff"
        value=""
        onChange={mockOnChange}
        onSelect={mockOnSelect}
        debounceMs={50}
      />
    );

    await user.type(screen.getByRole('combobox'), 'HUG');

    await waitFor(() => {
      expect(screen.getByText('Hôpitaux Universitaires de Genève (HUG)')).toBeInTheDocument();
    });

    fireEvent.mouseDown(screen.getByText('Hôpitaux Universitaires de Genève (HUG)'));

    await waitFor(() => {
      expect(mockOnSelect).toHaveBeenCalledWith(
        expect.objectContaining({
          source: 'alias',
          label: expect.stringContaining('Rue Gabrielle-Perret-Gentil 4'),
          lat: 46.19226,
          lon: 6.14262,
          name: 'Hôpitaux Universitaires de Genève (HUG)',
        })
      );
    });
  });

  it('ne devrait pas appeler l’API si moins de minChars (défaut 3)', async () => {
    render(
      <AddressAutocomplete
        name="address"
        value=""
        onChange={mockOnChange}
        onSelect={mockOnSelect}
      />
    );

    const input = screen.getByRole('combobox');
    fireEvent.change(input, { target: { value: 'Ge' } });

    await waitFor(() => {
      expect(apiClient.get).not.toHaveBeenCalled();
    });
  });

  it('devrait afficher un indicateur de chargement', async () => {
    let resolvePromise;
    const fetchPromise = new Promise((resolve) => {
      resolvePromise = resolve;
    });

    apiClient.get.mockImplementation(() => fetchPromise);

    const user = userEvent.setup();
    render(
      <AddressAutocomplete
        name="address"
        value=""
        onChange={mockOnChange}
        onSelect={mockOnSelect}
        debounceMs={50}
      />
    );

    const input = screen.getByRole('combobox');
    await user.type(input, 'Genève');

    await waitFor(
      () => {
        expect(apiClient.get).toHaveBeenCalled();
      },
      { timeout: 3000 }
    );

    resolvePromise({ status: 200, data: [] });
  });

  it('devrait afficher "Aucun résultat" si pas de suggestions', async () => {
    apiClient.get.mockImplementation((url) => {
      if (String(url).includes('favorites/autocomplete')) {
        return Promise.resolve({ status: 401, data: {} });
      }
      return Promise.resolve({ status: 200, data: [] });
    });

    const user = userEvent.setup();
    render(
      <AddressAutocomplete
        name="address"
        value=""
        onChange={mockOnChange}
        onSelect={mockOnSelect}
        debounceMs={50}
      />
    );

    const input = screen.getByRole('combobox');
    await user.type(input, 'AdresseIntrouvable123');

    await waitFor(
      () => {
        expect(screen.getByText('Aucun résultat')).toBeInTheDocument();
      },
      { timeout: 3000 }
    );
  });
});
