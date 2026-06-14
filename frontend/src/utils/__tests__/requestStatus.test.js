import {
  isExternalRequest,
  isAssignedRequest,
  isCompletedRequest,
  isConvertedLirie,
  hasBooking,
  getRequestStatusLabel,
  getCarrierSourceLabel,
  canAssignExternalCarrier,
  canCompleteExternalMission,
} from '../requestStatus';

describe('requestStatus', () => {
  it('détecte une demande externe', () => {
    expect(isExternalRequest({ carrier_source: 'external' })).toBe(true);
    expect(isExternalRequest({ carrier_source: 'lirie' })).toBe(false);
  });

  it('détecte une mission externe assignée', () => {
    expect(isAssignedRequest({ status: 'EXTERNAL_ASSIGNED' })).toBe(true);
  });

  it('détecte une mission externe terminée', () => {
    expect(isCompletedRequest({ status: 'EXTERNAL_DECLARED_COMPLETED' })).toBe(true);
    expect(isCompletedRequest({
      booking_summary: { status: 'COMPLETED', completed_at: '2026-06-01T10:00:00Z' },
    })).toBe(true);
  });

  it('distingue conversion LIRIE et garde booking', () => {
    expect(isConvertedLirie({ status: 'CONVERTED', booking_id: 42 })).toBe(true);
    expect(hasBooking({ booking_id: 42, booking_summary: { id: 42 } })).toBe(true);
    expect(hasBooking({ booking_id: 42 })).toBe(false);
  });

  it('privilégie status_label API', () => {
    expect(getRequestStatusLabel({ status: 'DRAFT', status_label: 'Brouillon API' }))
      .toBe('Brouillon API');
  });

  it('libellé mode d\'exécution', () => {
    expect(getCarrierSourceLabel({ carrier_source: 'external', carrier_source_label: 'Externe' }))
      .toBe('Externe');
  });

  it('actions externes contextuelles', () => {
    expect(canAssignExternalCarrier({ status: 'DRAFT', carrier_source: 'lirie' })).toBe(true);
    expect(canAssignExternalCarrier({ status: 'SENT', carrier_source: 'lirie' })).toBe(true);
    expect(canAssignExternalCarrier({ status: 'DRAFT', carrier_source: 'external' })).toBe(false);
    expect(canCompleteExternalMission({
      status: 'EXTERNAL_ASSIGNED',
      carrier_source: 'external',
    })).toBe(true);
  });
});
