import { renderHook, waitFor } from '@testing-library/react';
import { useAssignmentActions } from '../useAssignmentActions';
import * as companyService from '../../services/companyService';

// Mock des services
jest.mock('../../services/companyService');

describe('useAssignmentActions', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should initialize with default values', () => {
    const { result } = renderHook(() => useAssignmentActions());

    expect(result.current.loading).toBe(false);
    expect(result.current.error).toBeNull();
    expect(result.current.success).toBeNull();
  });

  it('should assign driver successfully', async () => {
    companyService.assignDriver.mockResolvedValue({ success: true });

    const { result } = renderHook(() => useAssignmentActions());

    const success = await result.current.handleAssignDriver(1, 10);

    await waitFor(() => {
      expect(success).toBe(true);
      expect(result.current.success).toBe('Chauffeur assigné avec succès');
      expect(result.current.error).toBeNull();
    });

    expect(companyService.assignDriver).toHaveBeenCalledWith(1, 10);
  });

  it('should handle assignment error', async () => {
    companyService.assignDriver.mockRejectedValue(new Error('Assignment failed'));

    const { result } = renderHook(() => useAssignmentActions());

    const success = await result.current.handleAssignDriver(1, 10);

    await waitFor(() => {
      expect(success).toBe(false);
      expect(result.current.error).toContain('Assignment failed');
      expect(result.current.success).toBeNull();
    });
  });

  it('should delete reservation successfully', async () => {
    companyService.deleteReservation.mockResolvedValue({ success: true });

    const { result } = renderHook(() => useAssignmentActions());

    const success = await result.current.handleDeleteReservation(5);

    await waitFor(() => {
      expect(success).toBe(true);
      expect(result.current.success).toBe('Réservation supprimée avec succès');
    });

    expect(companyService.deleteReservation).toHaveBeenCalledWith(5);
  });

  it('should handle deletion error', async () => {
    companyService.deleteReservation.mockRejectedValue(new Error('Deletion failed'));

    const { result } = renderHook(() => useAssignmentActions());

    const success = await result.current.handleDeleteReservation(5);

    await waitFor(() => {
      expect(success).toBe(false);
      expect(result.current.error).toContain('Deletion failed');
    });
  });

  it('should set loading state during operations', async () => {
    let resolveAssign;
    const assignPromise = new Promise((resolve) => {
      resolveAssign = resolve;
    });

    companyService.assignDriver.mockImplementation(() => assignPromise);

    const { result } = renderHook(() => useAssignmentActions());

    // Démarrer l'assignation
    result.current.handleAssignDriver(1, 10);

    // Attendre que loading soit true
    await waitFor(() => {
      expect(result.current.loading).toBe(true);
    });

    // Résoudre la promesse
    resolveAssign({ success: true });

    // Attendre que loading repasse à false
    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });
  });
});
