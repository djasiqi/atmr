import {
  compactInvoiceRoute,
  formatPreviewDayMonth,
  presentInvoiceLinesPreview,
} from '../invoiceLinesPreviewUi';

const lhaPreviewLines = [
  {
    preview_row_id: 'unit:single:45696',
    booking_id: 45696,
    booking_ids: [45696],
    unit_type: 'single',
    segments_count: 1,
    scheduled_at: '2026-08-02T08:00:00',
    patient_name: 'MARTIN Alice',
    description: 'Trajet LHA → HUG',
    amount_ht: 40,
  },
  {
    preview_row_id: 'unit:single:45697',
    booking_id: 45697,
    booking_ids: [45697],
    unit_type: 'single',
    segments_count: 1,
    scheduled_at: '2026-08-05T08:00:00',
    patient_name: 'MARTIN Alice',
    description: 'Trajet LHA → HUG',
    amount_ht: 40,
  },
  {
    preview_row_id: 'unit:single:45698',
    booking_id: 45698,
    booking_ids: [45698],
    unit_type: 'single',
    segments_count: 1,
    scheduled_at: '2026-08-10T08:00:00',
    patient_name: 'MARTIN Alice',
    description: 'Trajet LHA → HUG',
    amount_ht: 40,
  },
  {
    preview_row_id: 'unit:round_trip:45702:45703',
    booking_id: 45702,
    booking_ids: [45702, 45703],
    unit_type: 'round_trip',
    segments_count: 2,
    is_round_trip_leg: true,
    round_trip_partner_booking_id: 45703,
    round_trip_primary_amount_ht: 40,
    round_trip_partner_amount_ht: 40,
    scheduled_at: '2026-08-15T08:00:00',
    patient_name: 'KLEIN Arturo',
    description: 'Trajet LHA → HUG',
    round_trip_partner_description: 'Trajet HUG → LHA',
    amount_ht: 80,
  },
  {
    preview_row_id: 'unit:single:45704',
    booking_id: 45704,
    booking_ids: [45704],
    unit_type: 'single',
    segments_count: 1,
    scheduled_at: '2026-08-16T08:00:00',
    patient_name: 'DUPONT Marie',
    description: 'Trajet LHA → HUG',
    amount_ht: 40,
  },
  {
    preview_row_id: 'unit:single:45705',
    booking_id: 45705,
    booking_ids: [45705],
    unit_type: 'single',
    segments_count: 1,
    scheduled_at: '2026-08-18T08:00:00',
    patient_name: 'BARBEY Jacques',
    description: 'Trajet LHA → HUG',
    amount_ht: 40,
  },
  {
    preview_row_id: 'unit:single:45706',
    booking_id: 45706,
    booking_ids: [45706],
    unit_type: 'single',
    segments_count: 1,
    scheduled_at: '2026-08-18T14:00:00',
    patient_name: 'BARBEY Jacques',
    description: 'Trajet LHA → HUG',
    amount_ht: 40,
  },
];

describe('invoiceLinesPreviewUi', () => {
  it('compacte la date et le trajet', () => {
    expect(formatPreviewDayMonth('2026-08-02T08:00:00')).toBe('02.08');
    expect(compactInvoiceRoute('Trajet LHA → HUG')).toBe('LHA → HUG');
  });

  it('distingue 8 prestations métier et 7 lignes de facture après A/R', () => {
    const presented = presentInvoiceLinesPreview(lhaPreviewLines, {
      prestationCount: 8,
      totalHt: 320,
    });
    expect(presented.visualLineCount).toBe(7);
    expect(presented.prestationCount).toBe(8);
    expect(presented.totalHt).toBe(320);
    const klein = presented.rows.find((row) => row.patientName === 'KLEIN Arturo');
    expect(klein).toMatchObject({
      isRoundTrip: true,
      segmentsCount: 2,
      amountHt: 80,
      outboundBookingId: 45702,
      returnBookingId: 45703,
    });
  });
});
