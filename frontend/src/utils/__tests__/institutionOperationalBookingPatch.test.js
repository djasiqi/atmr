import {
  applyFlushedDestinationTimes,
  buildOperationalBookingPatch,
} from '../institutionOperationalBookingPatch';

describe('buildOperationalBookingPatch', () => {
  const accessForm = {
    customer_name: 'Charlotte CAVADINI',
    reason: '',
    pickup_floor: '',
    pickup_door_code: '',
    dropoff_floor: '',
    dropoff_door_code: '',
    pickup_access_notes: '',
    dropoff_access_notes: '',
    notes_medical: '',
    wheelchair_need: false,
    wheelchair_client_has: false,
    delivery_description: '',
  };

  it('envoie le RDV édité 13:00, pas l\'ancien 14:00', () => {
    const payload = buildOperationalBookingPatch({
      editVersion: 1,
      accessForm,
      pickupLocation: 'Chemin des Courbes 9, 1247, Anières',
      pickupTime: '13:15',
      missionDate: '2026-09-12',
      destinations: [
        {
          address: 'HUG, Genève',
          establishment: 'HUG',
          service: 'Radiologie',
          doctor: '',
          destinationType: 'medical',
          scheduled_time: '2026-09-12T13:00:00',
        },
      ],
      returnToInstitution: true,
      returnTime: '',
    });

    expect(payload.appointment_time).toBe('2026-09-12T13:00:00');
    expect(payload.leg_appointments).toEqual([
      { index: 0, scheduled_time: '2026-09-12T13:00:00' },
    ]);
    expect(payload.appointment_time).not.toBe('2026-09-12T14:00:00');
  });

  it('applique le flush DOM 13:00 même si l\'état React est encore 14:00', () => {
    const destinations = applyFlushedDestinationTimes(
      [
        {
          address: 'HUG, Genève',
          establishment: 'HUG',
          service: 'Radiologie',
          doctor: '',
          destinationType: 'medical',
          scheduled_time: '2026-09-12T14:00:00',
        },
      ],
      '2026-09-12',
      ['13:00'],
    );
    const payload = buildOperationalBookingPatch({
      editVersion: 1,
      accessForm,
      pickupLocation: 'Chemin des Courbes 9',
      pickupTime: '13:15',
      missionDate: '2026-09-12',
      destinations,
      returnToInstitution: false,
    });

    expect(payload.appointment_time).toBe('2026-09-12T13:00:00');
    expect(payload.leg_appointments[0].scheduled_time).toBe('2026-09-12T13:00:00');
  });
});
