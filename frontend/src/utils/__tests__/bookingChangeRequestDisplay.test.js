import {
  buildChangeRequestDetailLines,
  extractChangedFieldKeys,
  formatChangedFieldLabels,
  formatChangeRequestExpiry,
  formatFieldChangeLine,
  formatSnapshotValue,
  summarizeBookingChangeRequest,
} from '../bookingChangeRequestDisplay';

describe('bookingChangeRequestDisplay', () => {
  it('extrait les clés depuis un objet changed_fields', () => {
    expect(extractChangedFieldKeys({ scheduled_time: true, pickup_location: false })).toEqual([
      'scheduled_time',
    ]);
  });

  it('formate les libellés FR connus', () => {
    expect(formatChangedFieldLabels(['scheduled_time', 'pickup_location'])).toEqual([
      'Horaire prévu',
      'Lieu de prise en charge',
    ]);
    expect(formatChangedFieldLabels(['doctor_name', 'pickup_floor'])).toEqual([
      'Médecin',
      'Étage départ',
    ]);
  });

  it('formate les valeurs de snapshot', () => {
    expect(formatSnapshotValue('doctor_name', 'Dr Martin')).toBe('Dr Martin');
    expect(formatSnapshotValue('pickup_floor', '')).toBeNull();
    expect(formatSnapshotValue('wheelchair_need', true)).toBe('Oui');
    expect(formatSnapshotValue('scheduled_time', '2026-06-16T14:30:00Z')).toMatch(/16\/06 1[46]:30/);
  });

  it('détaille une modification avant → après', () => {
    expect(
      formatFieldChangeLine('doctor_name', 'Médecin', 'Dr Martin', 'Dr Dupont'),
    ).toBe('Médecin : Dr Martin → Dr Dupont');
    expect(
      formatFieldChangeLine('pickup_door_code', 'Code porte départ', null, 'A1234'),
    ).toBe('Code porte départ : ajout « A1234 »');
    expect(
      formatFieldChangeLine('notes_medical', 'Notes médicales', 'Allergie latex', null),
    ).toBe('Notes médicales : suppression « Allergie latex »');
  });

  it('construit les lignes détaillées depuis les snapshots', () => {
    const lines = buildChangeRequestDetailLines({
      changed_fields: {
        doctor_name: true,
        pickup_floor: true,
        notes_medical: true,
        pickup_door_code: true,
        dropoff_access_notes: true,
      },
      before_snapshot: {
        doctor_name: 'Dr Martin',
        pickup_floor: '2',
        notes_medical: 'Allergie latex',
        pickup_door_code: null,
        dropoff_access_notes: 'Entrée B',
      },
      after_snapshot: {
        doctor_name: 'Dr Dupont',
        pickup_floor: '3',
        notes_medical: 'Allergie latex, fauteuil pliant',
        pickup_door_code: '4587',
        dropoff_access_notes: 'Entrée C, accueil niveau 0',
      },
    });

    expect(lines).toEqual([
      { key: 'doctor_name', text: 'Médecin : Dr Martin → Dr Dupont' },
      { key: 'pickup_floor', text: 'Étage départ : 2 → 3' },
      { key: 'pickup_door_code', text: 'Code porte départ : ajout « 4587 »' },
      { key: 'notes_medical', text: 'Notes médicales : Allergie latex → Allergie latex, fauteuil pliant' },
      { key: 'dropoff_access_notes', text: 'Consignes arrivée : Entrée B → Entrée C, accueil niveau 0' },
    ]);
  });

  it('résume une demande avec champs et expiration', () => {
    const summary = summarizeBookingChangeRequest({
      changed_fields: { scheduled_time: true },
      before_snapshot: { scheduled_time: '2026-06-16T08:00:00Z' },
      after_snapshot: { scheduled_time: '2026-06-16T09:00:00Z' },
      reason: 'Changement horaire RDV',
      expires_at: '2026-06-16T14:30:00Z',
    });
    expect(summary.changeLines[0].text).toMatch(/Horaire prévu : .* → .*/);
    expect(summary.reason).toBe('Changement horaire RDV');
    expect(summary.expiresAt).toBe('2026-06-16T14:30:00Z');
  });

  it('formate une date d\'expiration', () => {
    const label = formatChangeRequestExpiry('2026-06-16T14:30:00Z');
    expect(label).toMatch(/16\/06/);
    expect(label).toMatch(/\d{2}:\d{2}/);
  });
});
