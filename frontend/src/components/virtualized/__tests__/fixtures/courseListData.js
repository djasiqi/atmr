/**
 * Fixtures de données pour les tests de VirtualizedCourseList
 */

export const mockAssignments = [
  {
    id: 1,
    pickup: 'Hôpital de Genève',
    dropoff: 'Aéroport de Genève',
    scheduled_time: '2025-01-14T10:00:00Z',
  },
  {
    id: 2,
    pickup: 'Clinique de Lausanne',
    dropoff: 'Gare de Lausanne',
    scheduled_time: '2025-01-14T11:30:00Z',
  },
  {
    id: 3,
    pickup: 'Centre médical de Montreux',
    dropoff: 'Hôpital de Vevey',
    scheduled_time: '2025-01-14T14:00:00Z',
  },
];

export const mockAssignmentsLarge = Array.from({ length: 100 }, (_, i) => ({
  id: i + 1,
  pickup: `Pickup ${i + 1}`,
  dropoff: `Dropoff ${i + 1}`,
  scheduled_time: `2025-01-14T${10 + Math.floor(i / 10)}:${(i % 10) * 6}:00Z`,
}));

export const mockAssignmentsEmpty = [];

