import {
  buildMultiStopLegsPreview,
  buildMultiStopPayloadStops,
  filterValidMultiStopDestinations,
} from '../buildMultiStopLegsPreview';

const ORIGIN = 'Institution';

describe('buildMultiStopLegsPreview', () => {
  it('construit la chaîne complète avec retour institution', () => {
    const validStops = [
      { dropoff_location: 'HUG' },
      { dropoff_location: 'Clinique des Grangettes' },
    ];
    const legs = buildMultiStopLegsPreview({
      pickupLocation: ORIGIN,
      validStops,
      returnToInstitution: true,
    });

    expect(legs).toEqual([
      { sequence: 1, from: ORIGIN, to: 'HUG', isReturn: false },
      { sequence: 2, from: 'HUG', to: 'Clinique des Grangettes', isReturn: false },
      { sequence: 3, from: 'Clinique des Grangettes', to: ORIGIN, isReturn: true },
    ]);
  });

  it('re-chaîne après suppression d une étape intermédiaire', () => {
    const validStops = [{ dropoff_location: 'HUG' }];
    const legs = buildMultiStopLegsPreview({
      pickupLocation: ORIGIN,
      validStops,
      returnToInstitution: true,
    });

    expect(legs).toHaveLength(2);
    expect(legs[0].to).toBe('HUG');
    expect(legs[1]).toMatchObject({ from: 'HUG', to: ORIGIN, isReturn: true });
  });

  it('reflete un reorder drag and drop via validStops reordonnees', () => {
    const validStops = [
      { dropoff_location: 'Clinique des Grangettes' },
      { dropoff_location: 'HUG' },
    ];
    const legs = buildMultiStopLegsPreview({
      pickupLocation: ORIGIN,
      validStops,
      returnToInstitution: true,
    });

    expect(legs[0].to).toBe('Clinique des Grangettes');
    expect(legs[1].from).toBe('Clinique des Grangettes');
    expect(legs[1].to).toBe('HUG');
  });

  it('ignore les étapes vides', () => {
    const validStops = filterValidMultiStopDestinations([
      { dropoff_location: '  ' },
      { dropoff_location: 'HUG' },
      { dropoff_location: '' },
    ]);

    expect(validStops).toHaveLength(1);
    expect(validStops[0].dropoff_location).toBe('HUG');
  });
});

describe('buildMultiStopPayloadStops', () => {
  it('émet sequence 1-based et scheduled_time optionnel', () => {
    const payloadStops = buildMultiStopPayloadStops([
      { dropoff_location: 'HUG', scheduled_time: '2026-06-11T09:00' },
      { dropoff_location: 'Clinique' },
    ]);

    expect(payloadStops[0].sequence).toBe(1);
    expect(payloadStops[0].dropoff_location).toBe('HUG');
    expect(payloadStops[0].scheduled_time).toBe('2026-06-11T09:00:00');
    expect(payloadStops[1].sequence).toBe(2);
    expect(payloadStops[1].scheduled_time).toBeUndefined();
  });

  it('inclut les détails (établissement / service / médecin) si renseignés', () => {
    const payloadStops = buildMultiStopPayloadStops([
      {
        dropoff_location: 'HUG',
        dropoff_establishment: 'HUG',
        dropoff_service: 'Radiologie',
        dropoff_doctor: 'Dr. Martin',
      },
      { dropoff_location: 'Clinique', dropoff_service: '  ' },
    ]);

    expect(payloadStops[0]).toMatchObject({
      dropoff_establishment: 'HUG',
      dropoff_service: 'Radiologie',
      dropoff_doctor: 'Dr. Martin',
    });
    expect(payloadStops[1].dropoff_establishment).toBeUndefined();
    expect(payloadStops[1].dropoff_service).toBeUndefined();
  });
});
