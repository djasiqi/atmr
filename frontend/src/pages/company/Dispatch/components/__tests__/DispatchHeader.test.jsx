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
    modeLoading: false,
    styles: {},
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should render without crashing', () => {
    render(<DispatchHeader {...defaultProps} />);
    expect(screen.getByText('Dispatch')).toBeInTheDocument();
    expect(screen.getByText('Semi-Auto')).toBeInTheDocument();
  });

  it('ne montre pas les contrôles Semi-Auto pendant le chargement du mode', () => {
    render(
      <DispatchHeader {...defaultProps} dispatchMode={null} modeLoading />
    );

    expect(screen.getByText(/Chargement du mode de dispatch/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Chargement du mode de dispatch/i)).toBeInTheDocument();
    expect(screen.queryByText(/Lancer Dispatch/i)).not.toBeInTheDocument();
    expect(
      screen.queryByLabelText(/Chauffeurs reguliers prioritaires/i)
    ).not.toBeInTheDocument();
  });

  it('ne montre pas les contrôles Semi-Auto si mode null sans loading', () => {
    render(<DispatchHeader {...defaultProps} dispatchMode={null} modeLoading={false} />);
    expect(screen.queryByText(/Lancer Dispatch/i)).not.toBeInTheDocument();
  });

  it('should call onRunDispatch when dispatch button is clicked', () => {
    render(<DispatchHeader {...defaultProps} />);
    const dispatchButton = screen.getByText(/lancer/i);

    fireEvent.click(dispatchButton);

    expect(defaultProps.onRunDispatch).toHaveBeenCalled();
  });

  it('should disable dispatch button when loading', () => {
    render(<DispatchHeader {...defaultProps} loading />);
    const dispatchButton = screen.getByRole('button', { name: /en cours/i });

    expect(dispatchButton).toBeDisabled();
  });

  it('should toggle regularFirst checkbox', () => {
    render(<DispatchHeader {...defaultProps} />);
    const checkbox = screen.getByLabelText(/Chauffeurs reguliers prioritaires/i);

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
    expect(screen.getByText('Manuel')).toBeInTheDocument();
    expect(screen.getByText(/Mode manuel/i)).toBeInTheDocument();
    expect(screen.queryByText(/Lancer Dispatch/i)).not.toBeInTheDocument();

    rerender(<DispatchHeader {...defaultProps} dispatchMode="fully_auto" />);
    expect(screen.getByText('Automatique')).toBeInTheDocument();
  });
});
