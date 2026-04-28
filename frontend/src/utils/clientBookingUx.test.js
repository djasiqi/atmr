import { getClientBookingUx, normalizeClientBookingStatus } from './clientBookingUx';

describe('clientBookingUx', () => {
  it.each([
    [
      'awaiting_client_payment',
      'awaiting_payment',
      ['Annuler'],
    ],
    ['pending', 'pending', ['Voir', 'Annuler']],
    ['requested', 'pending', ['Voir', 'Annuler']],
    ['confirmed', 'confirmed', ['Voir', 'Modifier', 'Annuler']],
    ['assigned', 'confirmed', ['Voir', 'Modifier', 'Annuler']],
    ['driver_on_the_way', 'driver_on_the_way', ['Suivre']],
    ['in_progress', 'in_progress', ['Suivre']],
    ['completed', 'completed', ['Recommander']],
    ['canceled', 'cancelled', ['Recommander']],
    ['cancelled', 'cancelled', ['Recommander']],
    ['unknown_backend_state', 'unknown', ['Rafraîchir']],
  ])('normalise %s et expose les actions attendues', (input, expectedStatus, expectedActions) => {
    const normalized = normalizeClientBookingStatus(input);
    const ux = getClientBookingUx(input);
    expect(normalized).toBe(expectedStatus);
    expect(ux.status).toBe(expectedStatus);
    expect(ux.actions).toEqual(expectedActions);
  });
});

