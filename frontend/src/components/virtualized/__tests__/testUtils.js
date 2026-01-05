/**
 * Utilitaires de test pour les composants virtualisés
 */

import React from 'react';

/**
 * Mock de react-window pour les tests
 * 
 * react-window nécessite un conteneur DOM avec des dimensions réelles.
 * Ce mock simule le comportement en rendant tous les éléments.
 */
export const mockReactWindow = () => {
  jest.mock('react-window', () => ({
    VariableSizeList: ({ children, itemCount, itemData: _itemData, height, width }) => {
      const items = Array.from({ length: itemCount }, (_, i) => {
        const child = children({ index: i, style: { height: 50, top: i * 50 } });
        return React.cloneElement(child, { key: i, 'data-testid': `virtualized-item-${i}` });
      });
      return (
        <div data-testid="virtualized-list" style={{ height, width }}>
          {items}
        </div>
      );
    },
    FixedSizeList: ({ children, itemCount, itemData: _itemData, height, width }) => {
      const items = Array.from({ length: itemCount }, (_, i) => {
        const child = children({ index: i, style: { height: 50, top: i * 50 } });
        return React.cloneElement(child, { key: i, 'data-testid': `virtualized-item-${i}` });
      });
      return (
        <div data-testid="virtualized-list" style={{ height, width }}>
          {items}
        </div>
      );
    },
  }));
};

/**
 * Mock des hooks personnalisés utilisés par les composants virtualisés
 */
export const mockHooks = () => {
  // Mock useCompanySocket
  jest.mock('hooks/useCompanySocket', () => ({
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

  // Mock useDispatchStatus
  jest.mock('hooks/useDispatchStatus', () => ({
    __esModule: true,
    default: jest.fn(() => ({
      status: 'idle',
      isDispatching: false,
      dispatchStatus: null,
    })),
  }));

  // Mock useDispatchDelays
  jest.mock('hooks/useDispatchDelays', () => ({
    __esModule: true,
    default: jest.fn(() => ({
      delays: {},
      fetchingDelays: false,
    })),
  }));
};

/**
 * Mock des services API
 */
export const mockServices = () => {
  // Mock driverService
  jest.mock('services/driverService', () => ({
    fetchDriverAssignments: jest.fn(),
  }));

  // Mock companyService
  jest.mock('services/companyService', () => ({
    fetchDispatchDelays: jest.fn(),
    runDispatchForDay: jest.fn(),
  }));
};

/**
 * Mock des composants Material-UI utilisés
 */
export const mockMaterialUI = () => {
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
        <select value={value} onChange={onChange}>{children}</select>
      ),
      MenuItem: ({ children, value }) => <option value={value}>{children}</option>,
      FormControl: ({ children }) => <div>{children}</div>,
      InputLabel: ({ children }) => <label>{children}</label>,
      Alert: ({ children, severity }) => (
        <div data-testid="alert" data-severity={severity}>{children}</div>
      ),
    };
  });
};

/**
 * Helper pour créer un wrapper avec QueryClient
 */
export const createTestWrapper = () => {
  const { QueryClient, QueryClientProvider } = require('@tanstack/react-query');
  const { BrowserRouter } = require('react-router-dom');
  
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false, cacheTime: 0 },
      mutations: { retry: false },
    },
  });

  return ({ children }) => (
    <BrowserRouter>
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    </BrowserRouter>
  );
};

