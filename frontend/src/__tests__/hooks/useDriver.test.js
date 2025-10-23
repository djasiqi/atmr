// frontend/tests/hooks/useDriver.test.js
import { renderHook, waitFor, act } from '@testing-library/react';
import useDriver from 'hooks/useDriver';
import {
  fetchCompanyDriver,
  updateDriverStatus,
  deleteDriver,
} from 'services/companyService';

// Mocks
jest.mock('services/companyService');

describe('useDriver', () => {
  const mockDrivers = [
    {
      id: 1,
      user: { first_name: 'Pierre', last_name: 'Martin' },
      is_available: true,
      is_active: true,
      vehicle_type: 'berline',
    },
    {
      id: 2,
      user: { first_name: 'Marie', last_name: 'Dubois' },
      is_available: false,
      is_active: true,
      vehicle_type: 'ambulance',
    },
  ];

  beforeEach(() => {
    jest.clearAllMocks();
    jest.spyOn(console, 'error').mockImplementation();

    fetchCompanyDriver.mockResolvedValue(mockDrivers);
    updateDriverStatus.mockResolvedValue({ success: true });
    deleteDriver.mockResolvedValue({ success: true });
  });

  it('devrait charger les chauffeurs au montage', async () => {
    const { result } = renderHook(() => useDriver());

    expect(result.current.loading).toBe(true);

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.drivers).toEqual(mockDrivers);
    expect(fetchCompanyDriver).toHaveBeenCalled();
  });

  it("devrait mettre à jour le statut d'un chauffeur", async () => {
    const { result } = renderHook(() => useDriver());

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    // Changer le statut du chauffeur 1
    await act(async () => {
      await result.current.toggleDriverStatus(1, false);
    });

    expect(updateDriverStatus).toHaveBeenCalledWith(1, false);

    // Vérifier que l'état local est mis à jour
    const updatedDriver = result.current.drivers.find((d) => d.id === 1);
    expect(updatedDriver.is_active).toBe(false);
  });

  it('devrait supprimer un chauffeur', async () => {
    const { result } = renderHook(() => useDriver());

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.drivers).toHaveLength(2);

    // Supprimer le chauffeur 2
    await act(async () => {
      await result.current.deleteDriverById(2);
    });

    expect(deleteDriver).toHaveBeenCalledWith(2);
    expect(result.current.drivers).toHaveLength(1);
    expect(result.current.drivers.find((d) => d.id === 2)).toBeUndefined();
  });

  it('devrait gérer les erreurs de chargement', async () => {
    fetchCompanyDriver.mockRejectedValue(new Error('Network error'));

    const { result } = renderHook(() => useDriver());

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.error).toBe('Erreur lors du chargement des chauffeurs.');
    expect(console.error).toHaveBeenCalled();
  });

  it('devrait permettre de rafraîchir les chauffeurs', async () => {
    const { result } = renderHook(() => useDriver());

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    // Premier appel au montage
    expect(fetchCompanyDriver).toHaveBeenCalledTimes(1);

    // Rafraîchir
    await act(async () => {
      await result.current.refreshDrivers();
    });

    expect(fetchCompanyDriver).toHaveBeenCalledTimes(2);
  });

  it('devrait gérer un tableau vide de chauffeurs', async () => {
    fetchCompanyDriver.mockResolvedValue([]);

    const { result } = renderHook(() => useDriver());

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.drivers).toEqual([]);
  });

  it('devrait gérer les erreurs de mise à jour de statut', async () => {
    updateDriverStatus.mockRejectedValue(new Error('Update failed'));

    const { result } = renderHook(() => useDriver());

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    // Tenter de mettre à jour
    await act(async () => {
      await result.current.toggleDriverStatus(1, false);
    });

    expect(console.error).toHaveBeenCalledWith(
      'Erreur lors de la mise à jour du statut :',
      expect.any(Error)
    );
  });

  it('devrait gérer les erreurs de suppression', async () => {
    deleteDriver.mockRejectedValue(new Error('Delete failed'));

    const { result } = renderHook(() => useDriver());

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    // Tenter de supprimer
    await act(async () => {
      await result.current.deleteDriverById(1);
    });

    expect(console.error).toHaveBeenCalledWith(
      'Erreur lors de la suppression :',
      expect.any(Error)
    );

    // Le chauffeur ne devrait pas être supprimé de l'état local
    expect(result.current.drivers).toHaveLength(2);
  });
});
