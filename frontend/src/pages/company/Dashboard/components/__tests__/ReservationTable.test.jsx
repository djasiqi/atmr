/**
 * Tests d'intégration pour ReservationTable (wrapper)
 */

import React from 'react';
import { render, screen } from '@testing-library/react';
import ReservationTable from '../ReservationTable';
import { mockDispatches } from '../../../../../components/virtualized/__tests__/fixtures/dispatchData';

// Mock VirtualizedReservationTable - utiliser l'alias Jest qui correspond au moduleNameMapper
jest.mock('components/virtualized/VirtualizedReservationTable', () => {
  const MockComponent = jest.fn(({ reservations, onRowClick, onAccept, onReject, onAssign, onEdit, onDelete, onSchedule, onDispatchNow, hideAssign, hideSchedule, hideUrgent, hideEdit, hideDelete }) => (
    <div data-testid="virtualized-reservation-table">
      <div data-testid="reservations-count">{reservations?.length || 0}</div>
      <div data-testid="has-on-row-click">{onRowClick ? 'yes' : 'no'}</div>
      <div data-testid="has-on-accept">{onAccept ? 'yes' : 'no'}</div>
      <div data-testid="has-on-reject">{onReject ? 'yes' : 'no'}</div>
      <div data-testid="has-on-assign">{onAssign ? 'yes' : 'no'}</div>
      <div data-testid="has-on-edit">{onEdit ? 'yes' : 'no'}</div>
      <div data-testid="has-on-delete">{onDelete ? 'yes' : 'no'}</div>
      <div data-testid="has-on-schedule">{onSchedule ? 'yes' : 'no'}</div>
      <div data-testid="has-on-dispatch">{onDispatchNow ? 'yes' : 'no'}</div>
      <div data-testid="hide-assign">{hideAssign ? 'yes' : 'no'}</div>
      <div data-testid="hide-schedule">{hideSchedule ? 'yes' : 'no'}</div>
      <div data-testid="hide-urgent">{hideUrgent ? 'yes' : 'no'}</div>
      <div data-testid="hide-edit">{hideEdit ? 'yes' : 'no'}</div>
      <div data-testid="hide-delete">{hideDelete ? 'yes' : 'no'}</div>
    </div>
  ));
  return {
    __esModule: true,
    default: MockComponent,
  };
});

describe('ReservationTable (Wrapper)', () => {
  const mockOnRowClick = jest.fn();
  const mockOnAccept = jest.fn();
  const mockOnReject = jest.fn();
  const mockOnAssign = jest.fn();
  const mockOnEdit = jest.fn();
  const mockOnDelete = jest.fn();
  const mockOnSchedule = jest.fn();
  const mockOnDispatchNow = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should render VirtualizedReservationTable', () => {
    render(<ReservationTable reservations={mockDispatches} />);
    
    expect(screen.getByTestId('virtualized-reservation-table')).toBeInTheDocument();
  });

  it('should pass all props correctly', () => {
    render(
      <ReservationTable
        reservations={mockDispatches}
        onRowClick={mockOnRowClick}
        onAccept={mockOnAccept}
        onReject={mockOnReject}
        onAssign={mockOnAssign}
        onEdit={mockOnEdit}
        onDelete={mockOnDelete}
        onSchedule={mockOnSchedule}
        onDispatchNow={mockOnDispatchNow}
        hideAssign={true}
        hideSchedule={false}
        hideUrgent={true}
        hideEdit={false}
        hideDelete={true}
      />
    );
    
    expect(screen.getByTestId('reservations-count')).toHaveTextContent('3');
    expect(screen.getByTestId('has-on-row-click')).toHaveTextContent('yes');
    expect(screen.getByTestId('has-on-accept')).toHaveTextContent('yes');
    expect(screen.getByTestId('has-on-reject')).toHaveTextContent('yes');
    expect(screen.getByTestId('has-on-assign')).toHaveTextContent('yes');
    expect(screen.getByTestId('has-on-edit')).toHaveTextContent('yes');
    expect(screen.getByTestId('has-on-delete')).toHaveTextContent('yes');
    expect(screen.getByTestId('has-on-schedule')).toHaveTextContent('yes');
    expect(screen.getByTestId('has-on-dispatch')).toHaveTextContent('yes');
    expect(screen.getByTestId('hide-assign')).toHaveTextContent('yes');
    expect(screen.getByTestId('hide-schedule')).toHaveTextContent('no');
    expect(screen.getByTestId('hide-urgent')).toHaveTextContent('yes');
    expect(screen.getByTestId('hide-edit')).toHaveTextContent('no');
    expect(screen.getByTestId('hide-delete')).toHaveTextContent('yes');
  });

  it('should use default values for optional props', () => {
    render(<ReservationTable reservations={mockDispatches} />);
    
    expect(screen.getByTestId('hide-assign')).toHaveTextContent('no'); // default false
    expect(screen.getByTestId('hide-schedule')).toHaveTextContent('no'); // default false
    expect(screen.getByTestId('hide-urgent')).toHaveTextContent('no'); // default false
    expect(screen.getByTestId('hide-edit')).toHaveTextContent('no'); // default false
    expect(screen.getByTestId('hide-delete')).toHaveTextContent('no'); // default false
  });

  it('should handle missing optional callbacks', () => {
    render(<ReservationTable reservations={mockDispatches} />);
    
    expect(screen.getByTestId('has-on-row-click')).toHaveTextContent('no');
    expect(screen.getByTestId('has-on-accept')).toHaveTextContent('no');
    expect(screen.getByTestId('has-on-reject')).toHaveTextContent('no');
  });
});

