import {
  groupBookingsForDisplay,
  controlStatusLabel,
  payerTypeLabel,
  billingIntentFromPayerType,
  buildBillingControlQueryParams,
  parseBillingControlApiError,
  isBookingEditable,
  adjustBillingControlSummary,
  applyOptimisticControlStatus,
  applyOptimisticControlOverrides,
  applyOptimisticPayerChange,
  applyOptimisticPayerMutation,
  mergeBookingControlFromMutation,
  pruneSyncedControlOverrides,
} from '../institutionBillingControlUi';

describe('institutionBillingControlUi', () => {
  const sampleItems = [
    {
      booking_id: 1,
      scheduled_time: '2026-09-02T10:00:00',
      patient: { display_name: 'Mme X' },
      segment_type: 'outbound',
      transport_company: { company_id: 10, display_name: 'Emmenez-moi' },
      payer: { type: 'clinic' },
      control: { effective_status: 'pending_review' },
      billing: { editable: true, locked: false, invoiced: false },
    },
    {
      booking_id: 2,
      scheduled_time: '2026-09-02T15:00:00',
      patient: { display_name: 'Mme X' },
      segment_type: 'return',
      transport_company: { company_id: 10, display_name: 'Emmenez-moi' },
      payer: { type: 'patient' },
      control: { effective_status: 'validated' },
      billing: { editable: true, locked: false, invoiced: false },
    },
    {
      booking_id: 3,
      scheduled_time: '2026-09-03T09:00:00',
      patient: { display_name: 'M. Y' },
      segment_type: 'outbound',
      transport_company: { company_id: 10, display_name: 'Emmenez-moi' },
      payer: { type: 'clinic' },
      control: { effective_status: 'anomaly' },
      billing: { editable: true, locked: false, invoiced: false },
    },
  ];

  it('U08 — regroupe visuellement par patient + date sans fusionner les bookings', () => {
    const groups = groupBookingsForDisplay(sampleItems);
    expect(groups).toHaveLength(2);
    expect(groups[0].items).toHaveLength(2);
    expect(groups[0].items.map((i) => i.booking_id)).toEqual([1, 2]);
    expect(groups[1].items).toHaveLength(1);
  });

  it('mappe les libellés payeur / statut / intent', () => {
    expect(payerTypeLabel('clinic')).toBe('Clinique');
    expect(controlStatusLabel('pending_review')).toBe('À vérifier');
    expect(controlStatusLabel('auto_released')).toBe('Libérée à échéance');
    expect(billingIntentFromPayerType('clinic')).toBe('institution');
  });

  it('construit les query params API', () => {
    expect(buildBillingControlQueryParams({
      period: '2026-09',
      control_status: 'validated',
      page: 2,
    })).toEqual({
      page: 2,
      page_size: 50,
      period: '2026-09',
      control_status: 'validated',
    });
  });

  it('U14/U15 — locked non éditable + message 409', () => {
    expect(isBookingEditable({ billing: { editable: false, locked: true } })).toBe(false);
    expect(parseBillingControlApiError({ response: { status: 409, data: { error: 'Verrouillé' } } }))
      .toBe('Verrouillé');
  });

  it('U16 — message 403', () => {
    expect(parseBillingControlApiError({ response: { status: 403 } }))
      .toMatch(/Accès refusé/i);
  });

  it('ajuste les compteurs localement pending → validated', () => {
    expect(adjustBillingControlSummary(
      { validated: 0, pending_review: 1, anomaly: 0 },
      'pending_review',
      'validated',
    )).toEqual({
      total: 0,
      payer_clinic: 0,
      payer_patient: 0,
      validated: 1,
      pending_review: 0,
      anomaly: 0,
    });
  });

  it('met à jour une ligne et le résumé sans toucher les autres', () => {
    const updated = applyOptimisticControlStatus(
      {
        items: sampleItems,
        summary: { total: 3, validated: 1, pending_review: 1, anomaly: 1 },
      },
      1,
      'validated',
    );
    expect(updated.items[0].control.effective_status).toBe('validated');
    expect(updated.items[1].control.effective_status).toBe('validated');
    expect(updated.summary.validated).toBe(2);
    expect(updated.summary.pending_review).toBe(0);
  });

  it('applique plusieurs overrides indépendants', () => {
    const updated = applyOptimisticControlOverrides(
      {
        items: sampleItems,
        summary: { total: 3, validated: 1, pending_review: 1, anomaly: 1 },
      },
      {
        1: { status: 'validated' },
        3: { status: 'validated' },
      },
    );
    expect(updated.items[0].control.effective_status).toBe('validated');
    expect(updated.items[2].control.effective_status).toBe('validated');
    expect(updated.summary.validated).toBe(3);
    expect(updated.summary.pending_review).toBe(0);
    expect(updated.summary.anomaly).toBe(0);
  });

  it('fusionne la réponse PATCH (control_status) dans le cache liste', () => {
    const updated = mergeBookingControlFromMutation(
      {
        items: sampleItems,
        summary: { total: 3, validated: 1, pending_review: 1, anomaly: 1 },
      },
      1,
      { control_status: 'validated', validated_by_display_name: 'Marc' },
    );
    expect(updated.items[0].control.effective_status).toBe('validated');
    expect(updated.items[0].control.validated_by_display_name).toBe('Marc');
  });

  it('retire les overrides déjà synchronisés ou absents de la liste', () => {
    expect(pruneSyncedControlOverrides(
      { 1: { status: 'validated' }, 2: { status: 'validated' } },
      sampleItems,
    )).toEqual({ 1: { status: 'validated' } });
  });

  it('met à jour le payeur et les compteurs Clinique / Patient', () => {
    const updated = applyOptimisticPayerChange(
      {
        items: sampleItems,
        summary: { total: 3, payer_clinic: 2, payer_patient: 1 },
      },
      1,
      'patient',
    );
    expect(updated.items[0].payer.type).toBe('patient');
    expect(updated.summary.payer_clinic).toBe(1);
    expect(updated.summary.payer_patient).toBe(2);
    expect(updated.items[1].payer.type).toBe('patient');
  });

  it('remet un booking validé à « À vérifier » lors d’un changement de payeur', () => {
    const updated = applyOptimisticPayerMutation(
      {
        items: sampleItems,
        summary: {
          total: 3,
          payer_clinic: 2,
          payer_patient: 1,
          validated: 1,
          pending_review: 1,
          anomaly: 1,
        },
      },
      2,
      'clinic',
    );
    expect(updated.items[1].payer.type).toBe('clinic');
    expect(updated.items[1].control.effective_status).toBe('pending_review');
    expect(updated.summary.validated).toBe(0);
    expect(updated.summary.pending_review).toBe(2);
    expect(updated.summary.payer_clinic).toBe(3);
    expect(updated.summary.payer_patient).toBe(0);
  });

  it('applique un override payeur indépendamment du statut', () => {
    const updated = applyOptimisticControlOverrides(
      {
        items: sampleItems,
        summary: { total: 3, payer_clinic: 2, payer_patient: 1, validated: 1, pending_review: 1, anomaly: 1 },
      },
      { 1: { payer: 'patient' } },
    );
    expect(updated.items[0].payer.type).toBe('patient');
    expect(updated.items[0].control.effective_status).toBe('pending_review');
    expect(updated.summary.payer_patient).toBe(2);
  });
});
