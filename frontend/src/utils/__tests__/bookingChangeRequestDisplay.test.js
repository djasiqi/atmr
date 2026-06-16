import {
  extractChangedFieldKeys,
  formatChangedFieldLabels,
  formatChangeRequestExpiry,
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
  });

  it('résume une demande avec champs et expiration', () => {
    const summary = summarizeBookingChangeRequest({
      changed_fields: { scheduled_time: true },
      reason: 'Changement horaire RDV',
      expires_at: '2026-06-16T14:30:00Z',
    });
    expect(summary.fieldLabels).toEqual(['Horaire prévu']);
    expect(summary.reason).toBe('Changement horaire RDV');
    expect(summary.expiresAt).toBe('2026-06-16T14:30:00Z');
  });

  it('formate une date d\'expiration', () => {
    const label = formatChangeRequestExpiry('2026-06-16T14:30:00Z');
    expect(label).toMatch(/16\/06/);
    expect(label).toMatch(/\d{2}:\d{2}/);
  });
});
