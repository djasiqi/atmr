/**
 * Tests d'intégration pour DispatchTable (wrapper)
 */

import React from 'react';
import { render, screen } from '@testing-library/react';
import DispatchTable from '../DispatchTable';
import { mockDispatches } from '../../../../components/virtualized/__tests__/fixtures/dispatchData';

// Mock VirtualizedDispatchTable pour vérifier que les props sont passées
// Utiliser le chemin relatif depuis le fichier de test
jest.mock('../../../../components/virtualized/VirtualizedDispatchTable', () => {
  const MockComponent = jest.fn(({ dispatches, reload, showPlanner, initialDispatchDay, initialRegularFirst, initialAllowEmergency, drivers, onReassign }) => (
    <div data-testid="virtualized-dispatch-table">
      <div data-testid="dispatches-count">{Array.isArray(dispatches) ? dispatches.length : Object.keys(dispatches || {}).length}</div>
      <div data-testid="has-reload">{reload ? 'yes' : 'no'}</div>
      <div data-testid="show-planner">{showPlanner ? 'yes' : 'no'}</div>
      <div data-testid="dispatch-day">{initialDispatchDay || 'none'}</div>
      <div data-testid="regular-first">{initialRegularFirst ? 'yes' : 'no'}</div>
      <div data-testid="allow-emergency">{initialAllowEmergency ? 'yes' : 'no'}</div>
      <div data-testid="drivers-count">{drivers?.length || 0}</div>
      <div data-testid="has-on-reassign">{onReassign ? 'yes' : 'no'}</div>
    </div>
  ));
  return {
    __esModule: true,
    default: MockComponent,
  };
});

describe('DispatchTable (Wrapper)', () => {
  const mockReload = jest.fn();
  const mockOnReassign = jest.fn();
  const mockDrivers = [
    { id: 1, username: 'driver1', name: 'Driver 1' },
    { id: 2, username: 'driver2', name: 'Driver 2' },
  ];

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should render VirtualizedDispatchTable', () => {
    render(<DispatchTable dispatches={mockDispatches} reload={mockReload} />);
    
    expect(screen.getByTestId('virtualized-dispatch-table')).toBeInTheDocument();
  });

  it('should pass all props correctly', () => {
    render(
      <DispatchTable
        dispatches={mockDispatches}
        reload={mockReload}
        showPlanner={true}
        initialDispatchDay="2025-01-14"
        initialRegularFirst={true}
        initialAllowEmergency={false}
        drivers={mockDrivers}
        onReassign={mockOnReassign}
      />
    );
    
    expect(screen.getByTestId('dispatches-count')).toHaveTextContent('3');
    expect(screen.getByTestId('has-reload')).toHaveTextContent('yes');
    expect(screen.getByTestId('show-planner')).toHaveTextContent('yes');
    expect(screen.getByTestId('dispatch-day')).toHaveTextContent('2025-01-14');
    expect(screen.getByTestId('regular-first')).toHaveTextContent('yes');
    expect(screen.getByTestId('allow-emergency')).toHaveTextContent('no');
    expect(screen.getByTestId('drivers-count')).toHaveTextContent('2');
    expect(screen.getByTestId('has-on-reassign')).toHaveTextContent('yes');
  });

  it('should use default values for optional props', () => {
    render(<DispatchTable dispatches={mockDispatches} reload={mockReload} />);
    
    expect(screen.getByTestId('show-planner')).toHaveTextContent('yes'); // default true
    expect(screen.getByTestId('regular-first')).toHaveTextContent('yes'); // default true
    expect(screen.getByTestId('allow-emergency')).toHaveTextContent('yes'); // default true
  });

  it('should handle missing optional props', () => {
    render(<DispatchTable dispatches={mockDispatches} />);
    
    expect(screen.getByTestId('has-reload')).toHaveTextContent('no');
    expect(screen.getByTestId('has-on-reassign')).toHaveTextContent('no');
    expect(screen.getByTestId('drivers-count')).toHaveTextContent('0');
  });
});

