/**
 * Fixtures de données pour les tests de VirtualizedDispatchTable
 */

export const mockDispatches = [
  {
    id: 1,
    client_name: 'Jean Dupont',
    client: { full_name: 'Jean Dupont' },
    scheduled_time: '2025-01-14T10:00:00Z',
    pickup_location: 'Hôpital de Genève',
    dropoff_location: 'Aéroport de Genève',
    driver_username: 'driver1',
    driver: { username: 'driver1', name: 'Jean Martin' },
    assignment: {
      id: 1,
      driver_id: 1,
      status: 'assigned',
    },
    status: 'assigned',
    is_return: false,
  },
  {
    id: 2,
    customer_name: 'Marie Durand',
    client: { full_name: 'Marie Durand' },
    scheduled_time: '2025-01-14T11:30:00Z',
    pickup_location: 'Clinique de Lausanne',
    dropoff_location: 'Gare de Lausanne',
    driver_username: 'driver2',
    driver: { username: 'driver2', name: 'Pierre Martin' },
    assignment: {
      id: 2,
      driver_id: 2,
      status: 'assigned',
    },
    status: 'assigned',
    is_return: false,
  },
  {
    id: 3,
    customer_name: 'Paul Martin',
    client: { full_name: 'Paul Martin' },
    scheduled_time: '2025-01-14T14:00:00Z',
    pickup_location: 'Centre médical',
    dropoff_location: 'Domicile',
    driver_username: null,
    driver: null,
    assignment: null,
    status: 'pending',
    is_return: false,
  },
];

export const mockDispatchesLarge = Array.from({ length: 200 }, (_, i) => ({
  id: i + 1,
  customer_name: `Client ${i + 1}`,
  client: { full_name: `Client ${i + 1}` },
  scheduled_time: `2025-01-14T${10 + Math.floor(i / 20)}:${(i % 20) * 3}:00Z`,
  pickup_location: `Pickup ${i + 1}`,
  dropoff_location: `Dropoff ${i + 1}`,
  driver_username: i % 3 === 0 ? `driver${i % 10}` : null,
  driver: i % 3 === 0 ? { username: `driver${i % 10}`, name: `Driver ${i % 10}` } : null,
  assignment: i % 3 === 0 ? {
    id: i + 1,
    driver_id: (i % 10) + 1,
    status: 'assigned',
  } : null,
  status: i % 3 === 0 ? 'assigned' : i % 3 === 1 ? 'pending' : 'completed',
  is_return: i % 5 === 0,
}));

export const mockDispatchesEmpty = [];

export const mockDrivers = [
  { id: 1, username: 'driver1', name: 'Jean Martin', status: 'available', is_available: true },
  { id: 2, username: 'driver2', name: 'Pierre Martin', status: 'available', is_available: true },
  { id: 3, username: 'driver3', name: 'Marie Martin', status: 'busy', is_available: false },
];

export const mockDelays = {
  1: {
    delay_minutes: 5,
    driver_name: 'Jean Martin',
    driver_phone: '+41 79 123 45 67',
    driver_vehicle: 'ABC-123',
  },
  2: {
    delay_minutes: 15,
    driver_name: 'Pierre Martin',
    driver_phone: '+41 79 234 56 78',
  },
};

