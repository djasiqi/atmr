/**
 * Fixtures de données pour les tests de VirtualizedReservationsList
 */

export const mockBookings = [
  {
    id: 1,
    scheduled_time: '2025-01-20T10:00:00Z',
    pickup_location: 'Hôpital de Genève',
    dropoff_location: 'Aéroport de Genève',
    company_name: 'Transport Médical SA',
    driver_name: 'Jean Dupont',
    amount: 150,
    status: 'pending',
    isCancelling: false,
  },
  {
    id: 2,
    scheduled_time: '2025-01-21T14:30:00Z',
    pickup_location: 'Clinique de Lausanne',
    dropoff_location: 'Gare de Lausanne',
    company_name: 'Transport Médical SA',
    driver_name: 'Marie Martin',
    amount: 120,
    status: 'accepted',
    isCancelling: false,
  },
  {
    id: 3,
    scheduled_time: '2025-01-15T09:00:00Z', // Date passée
    pickup_location: 'Centre médical',
    dropoff_location: 'Domicile',
    company_name: 'Transport Médical SA',
    driver_name: 'Pierre Durand',
    amount: 100,
    status: 'completed',
    isCancelling: false,
  },
];

export const mockBookingsLarge = Array.from({ length: 50 }, (_, i) => ({
  id: i + 1,
  scheduled_time: `2025-01-${15 + (i % 10)}T${10 + Math.floor(i / 5)}:00:00Z`,
  pickup_location: `Pickup ${i + 1}`,
  dropoff_location: `Dropoff ${i + 1}`,
  company_name: 'Transport Médical SA',
  driver_name: `Driver ${i + 1}`,
  amount: 100 + i * 10,
  status: i % 3 === 0 ? 'completed' : i % 3 === 1 ? 'pending' : 'accepted',
  isCancelling: false,
}));

export const mockBookingsEmpty = [];

