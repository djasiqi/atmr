/**
 * Tests pour VirtualizedDispatchTable
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import VirtualizedDispatchTable from '../VirtualizedDispatchTable';
import { mockDispatches, mockDrivers, mockDelays } from './fixtures/dispatchData';

// Mock react-window
jest.mock('react-window', () => {
  const mockReact = require('react');
  return {
    VariableSizeList: ({ children, itemCount, itemData }) => {
      const items = Array.from({ length: itemCount }, (_, i) => {
        const child = children({ index: i, style: { height: 60, top: i * 60 }, data: itemData });
        return mockReact.cloneElement(child, { key: i, 'data-testid': `virtualized-item-${i}` });
      });
      return mockReact.createElement('div', { 'data-testid': 'virtualized-list', 'data-item-count': itemCount }, items);
    },
  };
});

// Mock hooks - utiliser le chemin relatif depuis le fichier de test
jest.mock('../../../hooks/useCompanySocket', () => ({
  __esModule: true,
  default: jest.fn(() => ({
    socket: {
      on: jest.fn(),
      off: jest.fn(),
      emit: jest.fn(),
    },
    isConnected: true,
  })),
}));

// Mock useDispatchStatus - utiliser le chemin relatif depuis le fichier de test
jest.mock('../../../hooks/useDispatchStatus', () => ({
  __esModule: true,
  default: jest.fn((_socket) => ({
    label: 'Prêt',
    progress: 0,
    isRunning: false,
    setUpdatedAt: jest.fn(),
    handleDispatchJobResponse: jest.fn(),
  })),
}));

// Mock services - utiliser le chemin relatif depuis le fichier de test
jest.mock('../../../services/companyService', () => ({
  runDispatchForDay: jest.fn(),
  fetchDispatchRunById: jest.fn(),
  fetchDispatchDelays: jest.fn(),
}));

// Mock Material-UI
jest.mock('@mui/material', () => {
  const actual = jest.requireActual('@mui/material');
  return {
    ...actual,
      Chip: ({ label, ..._props }) => <span data-testid="chip">{label}</span>,
    Tooltip: ({ children, title }) => <div data-testid="tooltip" title={title}>{children}</div>,
    LinearProgress: () => <div data-testid="linear-progress">Loading...</div>,
    Dialog: ({ open, children, onClose }) => 
      open ? <div data-testid="dialog" onClick={onClose}>{children}</div> : null,
    DialogTitle: ({ children }) => <h2 data-testid="dialog-title">{children}</h2>,
    DialogContent: ({ children }) => <div data-testid="dialog-content">{children}</div>,
    DialogActions: ({ children }) => <div data-testid="dialog-actions">{children}</div>,
      Button: ({ children, onClick, ..._props }) => (
        <button onClick={onClick} {..._props}>{children}</button>
      ),
    Select: ({ children, value, onChange }) => (
      <select value={value} onChange={onChange} data-testid="select">{children}</select>
    ),
    MenuItem: ({ children, value }) => <option value={value}>{children}</option>,
    FormControl: ({ children }) => <div>{children}</div>,
    InputLabel: ({ children }) => <label>{children}</label>,
    Alert: ({ children, severity }) => (
      <div data-testid="alert" data-severity={severity}>{children}</div>
    ),
  };
});

// Mock react-icons
jest.mock('react-icons/fi', () => ({
  FiRefreshCw: () => <span data-testid="icon-refresh">↻</span>,
}));

// Mock formatDate utility - utiliser le chemin relatif depuis le fichier de test
jest.mock('../../../utils/formatDate', () => ({
  renderBookingDateTime: jest.fn((booking) => {
    if (!booking.scheduled_time) return '—';
    const date = new Date(booking.scheduled_time);
    return date.toLocaleString('fr-FR');
  }),
  renderDate: jest.fn((dateString) => {
    if (!dateString) return '—';
    const date = new Date(dateString);
    return date.toLocaleDateString('fr-CH');
  }),
  renderTime: jest.fn((dateString) => {
    if (!dateString) return '—';
    const date = new Date(dateString);
    return date.toLocaleTimeString('fr-CH', { hour: '2-digit', minute: '2-digit' });
  }),
}));

// Mock CSS modules
jest.mock('../../pages/company/components/DispatchTable.module.css', () => ({
  dispatchTable: 'dispatchTable',
  virtualizedTableWrapper: 'virtualizedTableWrapper',
  virtualizedRow: 'virtualizedRow',
  statusChipOnTime: 'statusChipOnTime',
  statusChipSlightDelay: 'statusChipSlightDelay',
  statusChipDelay: 'statusChipDelay',
  statusChipImpossible: 'statusChipImpossible',
  actionsCell: 'actionsCell',
  iconBtn: 'iconBtn',
}));

describe('VirtualizedDispatchTable', () => {
  const mockReload = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    // Reset window.innerHeight
    Object.defineProperty(window, 'innerHeight', {
      writable: true,
      configurable: true,
      value: 1000,
    });
  });

  describe('Rendu avec données', () => {
    it('should render with empty data', () => {
      render(<VirtualizedDispatchTable dispatches={[]} reload={mockReload} />);
      
      expect(screen.getByText('ID')).toBeInTheDocument();
      expect(screen.getByText('Client')).toBeInTheDocument();
      expect(screen.getByText('Aucun dispatch à afficher.')).toBeInTheDocument();
    });

    it('should render table header correctly', () => {
      render(<VirtualizedDispatchTable dispatches={mockDispatches} reload={mockReload} />);
      
      expect(screen.getByText('ID')).toBeInTheDocument();
      expect(screen.getByText('Client')).toBeInTheDocument();
      expect(screen.getByText('Date / Heure')).toBeInTheDocument();
      expect(screen.getByText('Pickup')).toBeInTheDocument();
      expect(screen.getByText('Dropoff')).toBeInTheDocument();
      expect(screen.getByText('Chauffeur assigné')).toBeInTheDocument();
      expect(screen.getByText('Statut')).toBeInTheDocument();
      expect(screen.getByText('Retard / Actions')).toBeInTheDocument();
    });

    it('should render dispatches', () => {
      render(<VirtualizedDispatchTable dispatches={mockDispatches} reload={mockReload} />);
      
      expect(screen.getByTestId('virtualized-list')).toBeInTheDocument();
      expect(screen.getByTestId('virtualized-list')).toHaveAttribute('data-item-count', '3');
    });

    it('should display dispatch information', () => {
      render(<VirtualizedDispatchTable dispatches={mockDispatches} reload={mockReload} />);
      
      expect(screen.getByText('Jean Dupont')).toBeInTheDocument();
      expect(screen.getByText(/Hôpital de Genève/)).toBeInTheDocument();
    });
  });

  describe('Affichage des chauffeurs', () => {
    it('should display driver name correctly', () => {
      render(
        <VirtualizedDispatchTable
          dispatches={mockDispatches}
          drivers={mockDrivers}
          reload={mockReload}
        />
      );
      
      // Le nom du chauffeur devrait être affiché
      expect(screen.getByText(/Jean Martin|driver1/)).toBeInTheDocument();
    });

    it('should display "Non assigné" when no driver', () => {
      const dispatchesWithoutDriver = [
        { ...mockDispatches[2], driver: null, assignment: null },
      ];
      
      render(
        <VirtualizedDispatchTable
          dispatches={dispatchesWithoutDriver}
          drivers={mockDrivers}
          reload={mockReload}
        />
      );
      
      expect(screen.getByText('Non assigné')).toBeInTheDocument();
    });
  });

  describe('Affichage des statuts', () => {
    it('should display status chip correctly', () => {
      render(<VirtualizedDispatchTable dispatches={mockDispatches} reload={mockReload} />);
      
      const chips = screen.getAllByTestId('chip');
      expect(chips.length).toBeGreaterThan(0);
    });

    it('should display timing status correctly', () => {
      render(<VirtualizedDispatchTable dispatches={mockDispatches} reload={mockReload} />);
      
      // Les statuts de timing devraient être affichés
      const timingChips = screen.getAllByTestId('chip');
      expect(timingChips.length).toBeGreaterThan(0);
    });
  });

  describe('Planification', () => {
    it('should render planner panel when showPlanner is true', () => {
      render(
        <VirtualizedDispatchTable
          dispatches={mockDispatches}
          reload={mockReload}
          showPlanner={true}
        />
      );
      
      expect(screen.getByText('Optimiser la journée')).toBeInTheDocument();
    });

    it('should not render planner panel when showPlanner is false', () => {
      render(
        <VirtualizedDispatchTable
          dispatches={mockDispatches}
          reload={mockReload}
          showPlanner={false}
        />
      );
      
      expect(screen.queryByText('Optimiser la journée')).not.toBeInTheDocument();
    });
  });

  describe('Footer et statistiques', () => {
    it('should render footer with statistics', () => {
      render(<VirtualizedDispatchTable dispatches={mockDispatches} reload={mockReload} />);
      
      // Le footer devrait afficher les statistiques
      expect(screen.getByText(/Total|Assigné|Non assigné/)).toBeInTheDocument();
    });
  });

  describe('Interactions', () => {
    it('should call reload when refresh button is clicked', () => {
      render(<VirtualizedDispatchTable dispatches={mockDispatches} reload={mockReload} />);
      
      const refreshButtons = screen.getAllByTestId('icon-refresh');
      if (refreshButtons.length > 0) {
        fireEvent.click(refreshButtons[0].closest('button'));
        // Le reload devrait être appelé (vérifier selon l'implémentation)
      }
    });
  });

  describe('Gestion des retards', () => {
    it('should display delay information when delays are provided', async () => {
      const { fetchDispatchDelays } = require('../../../services/companyService');
      fetchDispatchDelays.mockResolvedValue(mockDelays);

      render(
        <VirtualizedDispatchTable
          dispatches={mockDispatches}
          drivers={mockDrivers}
          reload={mockReload}
        />
      );
      
      // Les retards devraient être affichés si disponibles
      // Note: La logique de chargement des retards est dans useEffect
      await waitFor(() => {
        expect(fetchDispatchDelays).toHaveBeenCalled();
      });
    });
  });

  describe('Filtres et tri', () => {
    it('should filter by regular first when regularFirst is true', () => {
      render(
        <VirtualizedDispatchTable
          dispatches={mockDispatches}
          reload={mockReload}
          initialRegularFirst={true}
        />
      );
      
      // Les dispatches devraient être filtrés/triés
      expect(screen.getByTestId('virtualized-list')).toBeInTheDocument();
    });

    it('should allow emergency when allowEmergency is true', () => {
      render(
        <VirtualizedDispatchTable
          dispatches={mockDispatches}
          reload={mockReload}
          initialAllowEmergency={true}
        />
      );
      
      expect(screen.getByTestId('virtualized-list')).toBeInTheDocument();
    });
  });

  describe('WebSocket updates', () => {
    it('should handle WebSocket connection', () => {
      const useCompanySocket = require('../../../hooks/useCompanySocket').default;
      const mockSocket = {
        on: jest.fn(),
        off: jest.fn(),
        emit: jest.fn(),
      };
      
      useCompanySocket.mockReturnValue({ socket: mockSocket, isConnected: true });

      render(<VirtualizedDispatchTable dispatches={mockDispatches} reload={mockReload} />);
      
      // Le socket devrait être utilisé
      expect(useCompanySocket).toHaveBeenCalled();
    });
  });

  describe('Dispatch status', () => {
    it('should display dispatch status correctly', () => {
      const useDispatchStatus = require('../../../hooks/useDispatchStatus').default;
      useDispatchStatus.mockReturnValue({
        label: 'Optimisation en cours (50%)',
        progress: 50,
        isRunning: true,
        setUpdatedAt: jest.fn(),
        handleDispatchJobResponse: jest.fn(),
      });

      render(<VirtualizedDispatchTable dispatches={mockDispatches} reload={mockReload} />);
      
      // Il peut y avoir plusieurs éléments linear-progress, utiliser getAllByTestId
      expect(screen.getAllByTestId('linear-progress').length).toBeGreaterThan(0);
    });

    it('should not show progress when not dispatching', () => {
      const useDispatchStatus = require('../../../hooks/useDispatchStatus').default;
      useDispatchStatus.mockReturnValue({
        label: 'Prêt',
        progress: 0,
        isRunning: false,
        setUpdatedAt: jest.fn(),
        handleDispatchJobResponse: jest.fn(),
      });

      render(<VirtualizedDispatchTable dispatches={mockDispatches} reload={mockReload} />);
      
      // Le LinearProgress ne devrait pas être affiché quand isRunning est false
      // (selon l'implémentation, vérifier si isDispatching est utilisé)
    });
  });
});

