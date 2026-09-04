import {
  groupBookingsForDisplay,
  controlStatusLabel,
  payerTypeLabel,
  billingIntentFromPayerType,
  buildBillingControlQueryParams,
  parseBillingControlApiError,
  isBookingEditable,
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
});
