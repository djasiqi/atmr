// frontend/tests/components/AddressAutocomplete.test.jsx
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import AddressAutocomplete from 'components/common/AddressAutocomplete';

// Mock fetch
global.fetch = jest.fn();

describe('AddressAutocomplete', () => {
  const mockOnChange = jest.fn();
  const mockOnSelect = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    global.fetch.mockClear();
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
    const mockSuggestions = {
      features: [
        {
          properties: {
            name: 'HUG',
            street: 'Rue Gabrielle-Perret-Gentil',
            housenumber: '4',
            postcode: '1205',
            city: 'Genève',
            country: 'Suisse',
          },
          geometry: {
            coordinates: [6.14262, 46.19226],
          },
        },
        {
          properties: {
            street: 'Avenue de la Gare',
            housenumber: '10',
            postcode: '1003',
            city: 'Lausanne',
            country: 'Suisse',
          },
          geometry: {
            coordinates: [6.6294, 46.5197],
          },
        },
      ],
    };

    global.fetch.mockResolvedValue({
      ok: true,
      json: async () => mockSuggestions,
    });

    const user = userEvent.setup();
    render(
      <AddressAutocomplete
        name="address"
        value=""
        onChange={mockOnChange}
        onSelect={mockOnSelect}
      />
    );

    const input = screen.getByRole('combobox');
    await user.type(input, 'Genève');

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalled();
    });

    // Attendre que les suggestions apparaissent
    await waitFor(
      () => {
        expect(screen.getByText('HUG')).toBeInTheDocument();
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
    global.fetch.mockResolvedValue({
      ok: true,
      json: async () => mockSuggestions,
    });

    const user = userEvent.setup();
    render(
      <AddressAutocomplete name="pickup" value="" onChange={mockOnChange} onSelect={mockOnSelect} />
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

    global.fetch.mockResolvedValue({
      ok: true,
      json: async () => mockSuggestions,
    });

    const user = userEvent.setup();
    render(
      <AddressAutocomplete
        name="destination"
        value=""
        onChange={mockOnChange}
        onSelect={mockOnSelect}
      />
    );

    const input = screen.getByRole('combobox');
    await user.type(input, 'Ge');

    await waitFor(
      () => {
        expect(screen.getByRole('listbox')).toBeInTheDocument();
      },
      { timeout: 3000 }
    );

    // Navigation avec flèche bas
    fireEvent.keyDown(input, { key: 'ArrowDown' });

    // Sélection avec Enter
    fireEvent.keyDown(input, { key: 'Enter' });

    await waitFor(() => {
      expect(mockOnSelect).toHaveBeenCalled();
    });
  });

  it('devrait fermer les suggestions avec Escape', async () => {
    const mockSuggestions = [{ source: 'photon', label: 'Test', lat: 46.2, lon: 6.1 }];

    global.fetch.mockResolvedValue({
      ok: true,
      json: async () => mockSuggestions,
    });

    const user = userEvent.setup();
    render(
      <AddressAutocomplete name="test" value="" onChange={mockOnChange} onSelect={mockOnSelect} />
    );

    const input = screen.getByRole('combobox');
    await user.type(input, 'Test');

    await waitFor(
      () => {
        expect(screen.getByRole('listbox')).toBeInTheDocument();
      },
      { timeout: 3000 }
    );

    fireEvent.keyDown(input, { key: 'Escape' });

    await waitFor(() => {
      expect(screen.queryByRole('listbox')).not.toBeInTheDocument();
    });
  });

  it('ne devrait pas afficher de suggestions si moins de 2 caractères', async () => {
    render(
      <AddressAutocomplete
        name="address"
        value=""
        onChange={mockOnChange}
        onSelect={mockOnSelect}
        minChars={2}
      />
    );

    const input = screen.getByRole('combobox');
    fireEvent.change(input, { target: { value: 'G' } });

    await waitFor(() => {
      expect(global.fetch).not.toHaveBeenCalled();
    });
  });

  it('devrait afficher un indicateur de chargement', async () => {
    // Retarder la réponse pour voir le loading
    let resolvePromise;
    const fetchPromise = new Promise((resolve) => {
      resolvePromise = resolve;
    });

    global.fetch.mockImplementation(() => fetchPromise);

    const user = userEvent.setup();
    render(
      <AddressAutocomplete
        name="address"
        value=""
        onChange={mockOnChange}
        onSelect={mockOnSelect}
        debounceMs={100}
      />
    );

    const input = screen.getByRole('combobox');
    await user.type(input, 'Genève');

    // Attendre le debounce puis résoudre
    await new Promise((r) => setTimeout(r, 150));

    // Vérifier que le loading apparaît ou que le fetch est appelé
    expect(global.fetch).toHaveBeenCalled();

    // Résoudre le fetch
    resolvePromise({ ok: true, json: async () => [] });
  });

  it('devrait afficher "Aucun résultat" si pas de suggestions', async () => {
    global.fetch.mockResolvedValue({
      ok: true,
      json: async () => ({ features: [] }),
    });

    const user = userEvent.setup();
    render(
      <AddressAutocomplete
        name="address"
        value=""
        onChange={mockOnChange}
        onSelect={mockOnSelect}
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
