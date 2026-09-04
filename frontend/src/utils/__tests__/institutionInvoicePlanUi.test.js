import {
  MANUAL_FALLBACK_PAYER_TYPES,
  autoPlanParityWithCertifiedBuckets,
  groupAuditRowsForDisplay,
  originLabel,
  payerLabel,
  presentAuditRows,
  presentAutoDetectedPlan,
  presentInstitutionBillableRows,
  presentInstitutionBillablePreviewLines,
  presentInstitutionExcludedRows,
  presentInstitutionAlreadyInvoicedRows,
  presentInstitutionInvoiceSummary,
  alreadyInvoicedInvoiceLabel,
  draftInvoicePlanParity,
  exclusionDisplayLabel,
  exclusionWhyText,
  excludedBlockTitle,
  excludedRowWhoLabel,
  validationLabel,
} from '../institutionInvoicePlanUi';

const lhaLikePlan = {
  clinic: {
    display_name: 'Clinique les Hauts d’Anières',
    transports_count: 8,
    estimated_total: 320,
  },
  patients: [
    {
      institution_patient_id: 11,
      display_name: 'Charlotte CAVADINI',
      transports_count: 1,
      estimated_total: 40,
      booking_ids: [101],
    },
    {
      institution_patient_id: 12,
      display_name: 'Marie DUPONT',
      transports_count: 1,
      estimated_total: 40,
      booking_ids: [102],
    },
  ],
  partners: [],
  eligibility: {
    origin: { own_portfolio: 1, market_lirie: 11 },
    market_lirie: {
      validated: 8,
      auto_released: 0,
      pending: 1,
      disputed: 1,
    },
  },
  reconciliation: {
    buckets: {
      clinic_billable: { count: 8, amount_ht: 320, booking_ids: [1, 2, 3, 4, 5, 6, 7, 8] },
      patient_billable: { count: 2, amount_ht: 80, booking_ids: [101, 102] },
      pending_blocked: { count: 1, amount_ht: 40, booking_ids: [201] },
      disputed_blocked: { count: 1, amount_ht: 40, booking_ids: [202] },
      already_invoiced: { count: 1, amount_ht: 40, booking_ids: [99] },
    },
    bookings: [
      {
        booking_id: 15,
        origin: 'LIRIE_MARKETPLACE',
        validation_status: 'validated',
        payer: 'clinic',
        eligible: true,
        invoice_bucket: 'clinic_billable',
        group_id: 'rt-klein',
        grouping_relation: 'parent_booking_id',
        amount_ht: 40,
        patient_name: 'Arturo KLEIN',
        scheduled_at: '2026-08-15T08:00:00',
        pickup_location: 'LHA',
        dropoff_location: 'HUG',
      },
      {
        booking_id: 16,
        origin: 'LIRIE_MARKETPLACE',
        validation_status: 'validated',
        payer: 'clinic',
        eligible: true,
        invoice_bucket: 'clinic_billable',
        group_id: 'rt-klein',
        grouping_relation: 'parent_booking_id',
        amount_ht: 40,
        patient_name: 'Arturo KLEIN',
        scheduled_at: '2026-08-15T10:00:00',
        pickup_location: 'HUG',
        dropoff_location: 'LHA',
      },
      {
        booking_id: 102,
        origin: 'LIRIE_MARKETPLACE',
        validation_status: 'validated',
        payer: 'patient',
        eligible: true,
        group_id: null,
        amount_ht: 40,
      },
      {
        booking_id: 201,
        origin: 'LIRIE_MARKETPLACE',
        validation_status: 'pending',
        payer: 'clinic',
        eligible: false,
        invoice_bucket: 'pending_blocked',
        amount_ht: 40,
      },
      {
        booking_id: 99,
        origin: 'OWN_PORTFOLIO',
        validation_status: 'validated',
        payer: 'clinic',
        eligible: false,
        invoice_bucket: 'already_invoiced',
        amount_ht: 40,
        invoice_line_id: 501,
        invoice_number: 'EM-2026-08-0003',
      },
    ],
  },
};

describe('institutionInvoicePlanUi — plan auto-détecté', () => {
  it('affiche clinique / patients / bloquées depuis le plan certifié', () => {
    const presented = presentAutoDetectedPlan(lhaLikePlan);
    expect(presented.clinic).toMatchObject({
      visible: true,
      transportsCount: 8,
      totalHt: 320,
    });
    expect(presented.clinic.bookingIds).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
    expect(presented.patients).toMatchObject({
      visible: true,
      patientCount: 2,
      transportsCount: 2,
      totalHt: 80,
    });
    expect(presented.patients.bookingIds).toEqual([101, 102]);
    expect(presented.partners.visible).toBe(false);
    expect(presented.blocked).toMatchObject({
      visible: true,
      billable: false,
      pendingCount: 1,
      disputedCount: 1,
    });
    expect(presented.blocked.pendingBookingIds).toEqual([201]);
    expect(presented.blocked.disputedBookingIds).toEqual([202]);
  });

  it('parité ancien/nouveau flux : mêmes totaux que les buckets', () => {
    expect(autoPlanParityWithCertifiedBuckets(lhaLikePlan)).toEqual({
      clinicCountMatch: true,
      clinicHtMatch: true,
      patientCountMatch: true,
      patientTransportsMatch: true,
      patientHtMatch: true,
      patientBookingIdsMatch: true,
      blockedNotBillable: true,
    });
  });

  it('conserve origine, validation, payeur et A/R dépliable', () => {
    expect(originLabel('OWN_PORTFOLIO')).toBe('Portefeuille');
    expect(originLabel('LIRIE_MARKETPLACE')).toBe('Market LIRIE');
    expect(validationLabel('pending')).toBe('En attente');
    expect(validationLabel('disputed')).toBe('Contestée');
    expect(payerLabel('clinic')).toBe('Clinique');
    const rows = presentAuditRows(lhaLikePlan);
    const groups = groupAuditRowsForDisplay(rows);
    const ar = groups.find((g) => g.key.startsWith('rt-klein'));
    expect(ar.expandable).toBe(true);
    expect(ar.rows.map((r) => r.bookingId)).toEqual([15, 16]);
    expect(ar.rows.every((r) => r.originLabel === 'Market LIRIE')).toBe(true);
    expect(ar.rows.every((r) => r.payerLabel === 'Clinique')).toBe(true);
    expect(rows.find((r) => r.bookingId === 201).eligible).toBe(false);
  });

  it('garde le fallback manuel sur les trois radios', () => {
    expect(MANUAL_FALLBACK_PAYER_TYPES).toEqual(['patient', 'clinic', 'partner']);
  });

  it('résumé Institution = payeur institution seulement, sans patients ni partenaires', () => {
    const summary = presentInstitutionInvoiceSummary(lhaLikePlan);
    expect(summary).toMatchObject({
      transportsCount: 8,
      totalHt: 320,
      hasBillable: true,
    });
    expect(summary.bookingIds).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
    expect(summary.alreadyInvoiced).toMatchObject({
      visible: true,
      count: 1,
      amountHt: 40,
    });
    expect(summary.alreadyInvoiced.bookingIds).toEqual([99]);
    expect(summary.excluded).toMatchObject({
      visible: true,
      count: 2,
    });
    expect(summary.excluded.bookingIds).toEqual([201, 202]);
    expect(summary.patients).toBeUndefined();
    expect(summary.partners).toBeUndefined();
    expect(summary.eligibility).toBeUndefined();
    const billable = presentInstitutionBillableRows(lhaLikePlan);
    expect(billable.every((row) => row.payer === 'clinic' && row.eligible)).toBe(true);
    expect(billable.map((row) => row.bookingId)).toEqual([15, 16]);
    const excluded = presentInstitutionExcludedRows(lhaLikePlan);
    expect(excluded.map((row) => row.bookingId)).toEqual([201]);
    expect(excluded.every((row) => !row.eligible)).toBe(true);
    expect(excluded.every((row) => row.invoiceBucket !== 'already_invoiced')).toBe(
      true
    );
    expect(exclusionDisplayLabel(excluded[0])).toBe('En attente de validation');
    expect(
      exclusionDisplayLabel({
        invoiceBucket: 'disputed_blocked',
        validationStatus: 'disputed',
      })
    ).toBe("Contestée par l'institution");
    expect(
      exclusionWhyText({
        invoiceBucket: 'disputed_blocked',
        validationStatus: 'disputed',
      })
    ).toContain('contesté');
    expect(
      exclusionWhyText({
        invoiceBucket: 'pending_blocked',
        validationStatus: 'pending',
      })
    ).toContain("n'est pas encore validée");
    expect(excludedBlockTitle(1)).toBe("Pourquoi cette course n'est pas facturée");
    expect(excludedBlockTitle(2)).toBe(
      'Pourquoi ces courses ne sont pas facturées'
    );
    expect(
      excludedRowWhoLabel({ patientName: 'Alice MARTIN', bookingId: 45700 })
    ).toBe('Alice MARTIN');
    expect(excludedRowWhoLabel({ bookingId: 45700 })).toBe('');
    const previewLines = presentInstitutionBillablePreviewLines(lhaLikePlan);
    expect(previewLines).toHaveLength(1);
    expect(previewLines[0]).toMatchObject({
      unit_type: 'round_trip',
      segments_count: 2,
      patient_name: 'Arturo KLEIN',
      amount_ht: 80,
    });
    expect(previewLines[0].booking_ids).toEqual([15, 16]);
    expect(previewLines.some((line) => line.booking_ids.includes(102))).toBe(false);
    expect(previewLines.some((line) => line.booking_ids.includes(201))).toBe(false);
    const arFromParent = presentInstitutionBillablePreviewLines({
      reconciliation: {
        bookings: [
          {
            booking_id: 45702,
            payer: 'clinic',
            eligible: true,
            invoice_bucket: 'clinic_billable',
            group_id: 'request_id:2271',
            amount_ht: 40,
            patient_name: 'Arturo KLEIN',
            scheduled_at: '2026-08-15T08:00:00',
          },
          {
            booking_id: 45703,
            payer: 'clinic',
            eligible: true,
            invoice_bucket: 'clinic_billable',
            group_id: 'parent_booking_id:45702',
            grouping_relation: 'parent_booking_id',
            amount_ht: 40,
            patient_name: 'Arturo KLEIN',
            scheduled_at: '2026-08-15T10:00:00',
          },
        ],
      },
    });
    expect(arFromParent).toHaveLength(1);
    expect(arFromParent[0].unit_type).toBe('round_trip');
    expect(arFromParent[0].booking_ids).toEqual([45702, 45703]);
    const already = presentInstitutionAlreadyInvoicedRows(lhaLikePlan);
    expect(already.map((row) => row.bookingId)).toEqual([99]);
    expect(alreadyInvoicedInvoiceLabel(already[0])).toBe('Facture n° EM-2026-08-0003');
  });

  it('UI-DRAFT-6 / 7 : parité brouillon ↔ dernier plan', () => {
    const match = draftInvoicePlanParity(
      {
        id: 1,
        total_ht: 320,
        booking_ids: [1, 2, 3, 4, 5, 6, 7, 8],
      },
      lhaLikePlan
    );
    expect(match.totalMatch).toBe(true);
    expect(match.bookingIdsMatch).toBe(true);
    const stale = draftInvoicePlanParity(
      {
        id: 1,
        total_ht: 280,
        booking_ids: [10, 15],
      },
      lhaLikePlan
    );
    expect(stale.totalMatch).toBe(false);
    expect(stale.bookingIdsMatch).toBe(false);
  });
});
