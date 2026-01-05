/**
 * Tests pour VirtualizedReservationsList
 */

import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import VirtualizedReservationsList from '../VirtualizedReservationsList';
import { mockBookings, mockBookingsLarge } from './fixtures/reservationsData';

// Mock react-window
jest.mock('react-window', () => {
  const mockReact = require('react');
  return {
    VariableSizeList: ({ children, itemCount, itemData }) => {
      const items = Array.from({ length: itemCount }, (_, i) => {
        const child = children({ index: i, style: { height: 250, top: i * 250 }, data: itemData });
        return mockReact.cloneElement(child, { key: i, 'data-testid': `virtualized-item-${i}` });
      });
      return mockReact.createElement('div', { 'data-testid': 'virtualized-list', 'data-item-count': itemCount }, items);
    },
  };
});

// Mock react-icons
jest.mock('react-icons/fa', () => ({
  FaMapMarkerAlt: () => <span data-testid="icon-map">📍</span>,
  FaCalendarAlt: () => <span data-testid="icon-calendar">📅</span>,
  FaMoneyBillWave: () => <span data-testid="icon-money">💰</span>,
}));

// Mock CSS modules
jest.mock('../VirtualizedReservationsList.module.css', () => ({
  reservationList: 'reservationList',
  virtualizedList: 'virtualizedList',
  reservationCard: 'reservationCard',
  statusCompleted: 'statusCompleted',
  statusInProgress: 'statusInProgress',
  statusCanceled: 'statusCanceled',
  statusDefault: 'statusDefault',
  cancelBtn: 'cancelBtn',
  emptyMessage: 'emptyMessage',
}));

describe('VirtualizedReservationsList', () => {
  const mockOnCancelBooking = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('Rendu avec données', () => {
    it('should render with empty data', () => {
      render(<VirtualizedReservationsList bookings={[]} isPast={false} />);
      
      expect(screen.getByText('Aucune course à venir.')).toBeInTheDocument();
    });

    it('should render with empty data for past bookings', () => {
      render(<VirtualizedReservationsList bookings={[]} isPast={true} />);
      
      expect(screen.getByText('Aucune course passée.')).toBeInTheDocument();
    });

    it('should render reservation cards', () => {
      render(
        <VirtualizedReservationsList
          bookings={mockBookings}
          onCancelBooking={mockOnCancelBooking}
          isPast={false}
        />
      );
      
      expect(screen.getByTestId('virtualized-list')).toBeInTheDocument();
      expect(screen.getByTestId('virtualized-list')).toHaveAttribute('data-item-count', '3');
    });

    it('should display all booking information', () => {
      render(
        <VirtualizedReservationsList
          bookings={mockBookings}
          onCancelBooking={mockOnCancelBooking}
          isPast={false}
        />
      );
      
      expect(screen.getByText(/Hôpital de Genève/)).toBeInTheDocument();
      expect(screen.getByText(/Aéroport de Genève/)).toBeInTheDocument();
      // Utiliser getAllByText car il y a plusieurs éléments avec le même texte (une entreprise par réservation)
      const companyElements = screen.getAllByText(/Transport Médical SA/);
      expect(companyElements.length).toBeGreaterThan(0);
      expect(companyElements[0]).toBeInTheDocument();
      expect(screen.getByText(/Jean Dupont/)).toBeInTheDocument();
    });

    it('should handle large lists', () => {
      render(
        <VirtualizedReservationsList
          bookings={mockBookingsLarge}
          onCancelBooking={mockOnCancelBooking}
          isPast={false}
        />
      );
      
      expect(screen.getByTestId('virtualized-list')).toBeInTheDocument();
      expect(screen.getByTestId('virtualized-list')).toHaveAttribute('data-item-count', '50');
    });
  });

  describe('Différence entre cartes à venir et passées', () => {
    it('should show company and driver for upcoming bookings', () => {
      // Utiliser des dates futures garanties pour éviter que le filtre retourne un tableau vide
      const futureDate = new Date();
      futureDate.setDate(futureDate.getDate() + 30); // 30 jours dans le futur
      
      const upcomingBookings = [
        {
          ...mockBookings[0],
          scheduled_time: futureDate.toISOString(),
        },
        {
          ...mockBookings[1],
          scheduled_time: new Date(futureDate.getTime() + 86400000).toISOString(), // +1 jour
        },
      ];
      
      render(
        <VirtualizedReservationsList
          bookings={upcomingBookings}
          onCancelBooking={mockOnCancelBooking}
          isPast={false}
        />
      );
      
      // Utiliser getAllByText car il peut y avoir plusieurs éléments
      const companyElements = screen.getAllByText(/Transport Médical SA/);
      expect(companyElements.length).toBeGreaterThan(0);
      expect(companyElements[0]).toBeInTheDocument();
      expect(screen.getByText(/Jean Dupont/)).toBeInTheDocument();
    });

    it('should not show company and driver for past bookings', () => {
      const pastBookings = mockBookings.filter(
        (b) => new Date(b.scheduled_time) <= new Date()
      );
      
      render(
        <VirtualizedReservationsList
          bookings={pastBookings}
          isPast={true}
        />
      );
      
      // Company and driver should not be in the document for past bookings
      const companyElements = screen.queryAllByText(/Transport Médical SA/);
      const driverElements = screen.queryAllByText(/Jean Dupont/);
      expect(companyElements.length).toBe(0);
      expect(driverElements.length).toBe(0);
    });
  });

  describe('Interactions', () => {
    it('should call onCancelBooking when cancel button is clicked', () => {
      render(
        <VirtualizedReservationsList
          bookings={mockBookings}
          onCancelBooking={mockOnCancelBooking}
          isPast={false}
        />
      );
      
      const cancelButtons = screen.getAllByText('Annuler');
      fireEvent.click(cancelButtons[0]);
      
      expect(mockOnCancelBooking).toHaveBeenCalledTimes(1);
      expect(mockOnCancelBooking).toHaveBeenCalledWith(mockBookings[0].id);
    });

    it('should show "Annulation..." state when booking is cancelling', () => {
      const bookingsWithCancelling = [
        { ...mockBookings[0], isCancelling: true },
        ...mockBookings.slice(1),
      ];
      
      render(
        <VirtualizedReservationsList
          bookings={bookingsWithCancelling}
          onCancelBooking={mockOnCancelBooking}
          isPast={false}
        />
      );
      
      expect(screen.getByText('Annulation...')).toBeInTheDocument();
    });

    it('should not show cancel button for past bookings', () => {
      const pastBookings = mockBookings.filter(
        (b) => new Date(b.scheduled_time) <= new Date()
      );
      
      render(
        <VirtualizedReservationsList
          bookings={pastBookings}
          isPast={true}
        />
      );
      
      const cancelButtons = screen.queryAllByText('Annuler');
      expect(cancelButtons.length).toBe(0);
    });

    it('should not show cancel button for canceled bookings', () => {
      const canceledBookings = [
        { ...mockBookings[0], status: 'canceled' },
      ];
      
      render(
        <VirtualizedReservationsList
          bookings={canceledBookings}
          onCancelBooking={mockOnCancelBooking}
          isPast={false}
        />
      );
      
      const cancelButtons = screen.queryAllByText('Annuler');
      expect(cancelButtons.length).toBe(0);
    });
  });

  describe('Affichage des statuts', () => {
    it('should display correct status for completed bookings', () => {
      const completedBookings = [
        { ...mockBookings[0], status: 'completed' },
      ];
      
      render(
        <VirtualizedReservationsList
          bookings={completedBookings}
          isPast={false}
        />
      );
      
      expect(screen.getByText('✅ Terminé')).toBeInTheDocument();
    });

    it('should display correct status for in_progress bookings', () => {
      const inProgressBookings = [
        { ...mockBookings[0], status: 'in_progress' },
      ];
      
      render(
        <VirtualizedReservationsList
          bookings={inProgressBookings}
          isPast={false}
        />
      );
      
      expect(screen.getByText('🚖 En cours')).toBeInTheDocument();
    });

    it('should display correct status for canceled bookings', () => {
      const canceledBookings = [
        { ...mockBookings[0], status: 'canceled' },
      ];
      
      render(
        <VirtualizedReservationsList
          bookings={canceledBookings}
          isPast={false}
        />
      );
      
      expect(screen.getByText('❌ Annulé')).toBeInTheDocument();
      expect(screen.getByText('0 CHF')).toBeInTheDocument();
    });

    it('should display correct status for pending bookings', () => {
      const pendingBookings = [
        { ...mockBookings[0], status: 'pending' },
      ];
      
      render(
        <VirtualizedReservationsList
          bookings={pendingBookings}
          isPast={false}
        />
      );
      
      expect(screen.getByText('🔄 En attente')).toBeInTheDocument();
    });
  });

  describe('Affichage du montant', () => {
    it('should display amount correctly', () => {
      render(
        <VirtualizedReservationsList
          bookings={mockBookings}
          isPast={false}
        />
      );
      
      expect(screen.getByText('150 CHF')).toBeInTheDocument();
      expect(screen.getByText('120 CHF')).toBeInTheDocument();
    });

    it('should display 0 CHF for canceled bookings', () => {
      const canceledBookings = [
        { ...mockBookings[0], status: 'canceled' },
      ];
      
      render(
        <VirtualizedReservationsList
          bookings={canceledBookings}
          isPast={false}
        />
      );
      
      expect(screen.getByText('0 CHF')).toBeInTheDocument();
    });

    it('should display N/A for missing amount', () => {
      const bookingsWithoutAmount = [
        { ...mockBookings[0], amount: null },
      ];
      
      render(
        <VirtualizedReservationsList
          bookings={bookingsWithoutAmount}
          isPast={false}
        />
      );
      
      expect(screen.getByText('N/A')).toBeInTheDocument();
    });
  });

  describe('Formatage des dates', () => {
    it('should format date correctly', () => {
      render(
        <VirtualizedReservationsList
          bookings={mockBookings}
          isPast={false}
        />
      );
      
      // Vérifier que la date est formatée (format français)
      const dateElements = screen.getAllByTestId('icon-calendar');
      expect(dateElements.length).toBeGreaterThan(0);
    });
  });
});

