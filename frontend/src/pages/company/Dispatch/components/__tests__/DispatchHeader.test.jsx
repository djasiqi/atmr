import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import DispatchHeader from '../DispatchHeader';

describe('DispatchHeader', () => {
  const defaultProps = {
    date: '2024-01-15',
    setDate: jest.fn(),
    regularFirst: true,
    setRegularFirst: jest.fn(),
    allowEmergency: true,
    setAllowEmergency: jest.fn(),
    onRunDispatch: jest.fn(),
    loading: false,
    dispatchSuccess: null,
    dispatchMode: 'semi_auto',
    styles: {},
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should render without crashing', () => {
    render(<DispatchHeader {...defaultProps} />);
    expect(screen.getByText(/Dispatch Semi-Automatique/i)).toBeInTheDocument();
  });

  it('should display the current date', () => {
    render(<DispatchHeader {...defaultProps} />);
    const dateInput = screen.getByDisplayValue('2024-01-15');
    expect(dateInput).toBeInTheDocument();
  });

  it('should call setDate when date is changed', () => {
    render(<DispatchHeader {...defaultProps} />);
    const dateInput = screen.getByDisplayValue('2024-01-15');

    fireEvent.change(dateInput, { target: { value: '2024-01-16' } });

    expect(defaultProps.setDate).toHaveBeenCalledWith('2024-01-16');
  });

  it('should call onRunDispatch when dispatch button is clicked', () => {
    render(<DispatchHeader {...defaultProps} />);
    const dispatchButton = screen.getByText(/lancer/i);

    fireEvent.click(dispatchButton);

    expect(defaultProps.onRunDispatch).toHaveBeenCalled();
  });

  it('should disable dispatch button when loading', () => {
    render(<DispatchHeader {...defaultProps} loading={true} />);
    const dispatchButton = screen.getByText(/en cours/i);

    expect(dispatchButton).toBeDisabled();
  });

  it('should toggle regularFirst checkbox', () => {
    render(<DispatchHeader {...defaultProps} />);
    const checkbox = screen.getByLabelText(/Chauffeurs réguliers prioritaires/i);

    fireEvent.click(checkbox);

    expect(defaultProps.setRegularFirst).toHaveBeenCalledWith(false);
  });

  it('should toggle allowEmergency checkbox', () => {
    render(<DispatchHeader {...defaultProps} />);
    const checkbox = screen.getByLabelText(/Autoriser chauffeurs d'urgence/i);

    fireEvent.click(checkbox);

    expect(defaultProps.setAllowEmergency).toHaveBeenCalledWith(false);
  });

  it('should display success message', () => {
    const successMessage = 'Dispatch terminé avec succès';
    render(<DispatchHeader {...defaultProps} dispatchSuccess={successMessage} />);

    expect(screen.getByText(successMessage)).toBeInTheDocument();
  });

  it('should show different text based on dispatch mode', () => {
    const { rerender } = render(<DispatchHeader {...defaultProps} dispatchMode="manual" />);
    expect(screen.getByText(/Dispatch Manuel/i)).toBeInTheDocument();
    expect(screen.getByText(/Mode actuel:/i)).toBeInTheDocument();

    rerender(<DispatchHeader {...defaultProps} dispatchMode="fully_auto" />);
    expect(screen.getByText(/Dispatch Automatique/i)).toBeInTheDocument();
  });
});
