import {
  PREPARATION_INVOICE_ID,
  applyPreparationGlobalDiscount,
  applyPreparationToPersistedInvoice,
  addPreparationCustomLine,
  buildPreparationInvoice,
  planPreparationReplay,
  removePreparationLine,
  updatePreparationLine,
} from '../invoicePreparationDraft';

const clinicLines = [
  {
    booking_id: 11,
    booking_ids: [11],
    scheduled_at: '2026-10-06T09:10:00',
    patient_name: 'Claire Bonvin',
    description: 'Chemin des Vignes 4 → Clinique',
    amount_ht: 72,
    already_invoiced: false,
  },
  {
    booking_id: 12,
    scheduled_at: '2026-10-08T11:00:00',
    patient_name: 'Marc Duval',
    description: 'Vesenaz → HUG',
    amount_ht: 95,
  },
];

const buildClinic = () =>
  buildPreparationInvoice({
    payerType: 'clinic',
    periodYear: 2026,
    periodMonth: 10,
    recipientName: "Clinique les Hauts d'Anières",
    previewLines: clinicLines,
  });

describe('brouillon de préparation', () => {
  test('institution + période : aperçu sans numéro ni identifiant comptable', () => {
    const invoice = buildClinic();
    expect(invoice.preparation).toBe(true);
    expect(invoice.id).toBe(PREPARATION_INVOICE_ID);
    expect(invoice.invoice_number).toBe('');
    expect(invoice.billing_strategy).toBe('s2_clinic_monthly');
    expect(invoice.client.institution_name).toBe("Clinique les Hauts d'Anières");
    expect(invoice.lines).toHaveLength(2);
    expect(invoice.lines[0].line_meta.patient_name).toBe('Claire Bonvin');
    expect(invoice.total_amount).toBe(167);
    expect(invoice.vat_total_amount).toBe(0);
    expect(planPreparationReplay(invoice, { lines: [] }).steps).toEqual([]);
  });

  it('reprend la TVA déjà calculée par l’aperçu de période', () => {
    const invoice = buildPreparationInvoice({
      payerType: 'patient',
      periodYear: 2026,
      periodMonth: 10,
      recipientName: 'Patient Fictif',
      estimatedSubtotalHt: 240,
      estimatedVatTotal: 18.45,
      previewLines: [
        { booking_id: 1, booking_ids: [1], amount_ht: 45, description: 'Simple' },
        {
          booking_id: 2,
          booking_ids: [2, 3],
          amount_ht: 90,
          is_round_trip_leg: true,
          round_trip_primary_amount_ht: 45,
          round_trip_partner_amount_ht: 45,
          description: 'Aller',
          round_trip_partner_description: 'Retour',
        },
        { booking_id: 4, booking_ids: [4], amount_ht: 105, description: 'Reste' },
      ],
    });

    expect(invoice.subtotal_amount).toBe(240);
    expect(invoice.vat_total_amount).toBe(18.45);
    expect(invoice.total_amount).toBe(258.45);
    expect(invoice.meta.vat.applicable).toBe(true);
  });

  it('conserve le total TTC déjà clos par le serveur', () => {
    const invoice = buildPreparationInvoice({
      payerType: 'patient',
      periodYear: 2026,
      periodMonth: 10,
      recipientName: 'Patient Fictif',
      estimatedSubtotalHt: 240,
      estimatedVatTotal: 18.5,
      estimatedTotalWithVat: 258.5,
      previewLines: [
        { booking_id: 1, booking_ids: [1], amount_ht: 45, description: 'Simple' },
        {
          booking_id: 2,
          booking_ids: [2, 3],
          amount_ht: 90,
          is_round_trip_leg: true,
          round_trip_primary_amount_ht: 45,
          round_trip_partner_amount_ht: 45,
          description: 'Aller',
        },
        { booking_id: 4, booking_ids: [4], amount_ht: 40, description: 'Validé' },
        { booking_id: 5, booking_ids: [5], amount_ht: 20, description: 'Second' },
        { booking_id: 6, booking_ids: [6], amount_ht: 45, description: 'Attente' },
      ],
    });

    expect(invoice.subtotal_amount).toBe(240);
    expect(invoice.vat_total_amount).toBe(18.5);
    expect(invoice.total_amount).toBe(258.5);
  });

  test('modifier une ligne, une remise et une ligne HT recalcule le total', () => {
    let invoice = buildClinic();
    invoice = updatePreparationLine(invoice, invoice.lines[0].id, {
      description: 'Trajet modifié',
      line_total: 80,
    });
    expect(invoice.lines[0].description).toBe('Trajet modifié');
    expect(invoice.total_amount).toBe(175);

    invoice = applyPreparationGlobalDiscount(invoice, {
      global_discount_percent: 10,
      global_discount_note: 'Accord',
    });
    expect(invoice.meta.global_discount.percent).toBe(10);
    expect(invoice.total_amount).toBe(157.5);

    invoice = addPreparationCustomLine(invoice, {
      description: 'Attente',
      line_total: 20,
      qty: 1,
      custom_mode: 'quantity',
    });
    expect(invoice.lines.some((line) => line.description === 'Attente')).toBe(true);
    expect(invoice.total_amount).toBe(177.5);
  });

  test('kept et excluded booking ids suivent la préparation', () => {
    const {
      keptBookingIdsFromPreparation,
      excludedBookingIdsFromPreparation,
    } = require('../invoicePreparationDraft');
    let invoice = buildClinic();
    expect(keptBookingIdsFromPreparation(invoice)).toEqual([11, 12]);
    invoice = removePreparationLine(invoice, invoice.lines[1].id);
    expect(keptBookingIdsFromPreparation(invoice)).toEqual([11]);
    expect(excludedBookingIdsFromPreparation(invoice)).toEqual([12]);
  });

  test('retirer une ligne la sort du plan de création', () => {
    let invoice = buildClinic();
    invoice = removePreparationLine(invoice, invoice.lines[1].id);
    const server = {
      lines: [
        { id: 501, reservation_id: 11, description: clinicLines[0].description, line_total: 72 },
        { id: 502, reservation_id: 12, description: clinicLines[1].description, line_total: 95 },
      ],
    };
    const plan = planPreparationReplay(invoice, server);
    expect(plan.steps.map((step) => step.type)).toEqual(['remove']);
    expect(plan.steps[0].lineId).toBe(502);
  });

  test('la création rejoue l’état préparé une seule fois', async () => {
    let invoice = buildClinic();
    invoice = updatePreparationLine(invoice, invoice.lines[0].id, { line_total: 70 });
    invoice = applyPreparationGlobalDiscount(invoice, { global_discount_percent: 10 });
    const server = {
      id: 90,
      invoice_number: 'EM-2026-10-0008',
      lines: [
        { id: 1, reservation_id: 11, description: clinicLines[0].description, line_total: 72 },
        { id: 2, reservation_id: 12, description: clinicLines[1].description, line_total: 95 },
      ],
    };
    const calls = [];
    const adapter = {
      async updateLine(lineId, body) {
        calls.push(['update', lineId, body.line_total]);
        return { invoice: server };
      },
      async applyGlobalDiscount(body) {
        calls.push(['discount', body.global_discount_percent]);
        return { invoice: server };
      },
      async removeLine() {
        calls.push(['remove']);
        return { invoice: server };
      },
      async addCustomLine() {
        calls.push(['add']);
        return { invoice: server };
      },
      async load() {
        calls.push(['load']);
        return { ...server, invoice_number: 'EM-2026-10-0008' };
      },
    };
    const persisted = await applyPreparationToPersistedInvoice(adapter, server, invoice);
    expect(persisted.invoice_number).toBe('EM-2026-10-0008');
    expect(calls.filter((row) => row[0] === 'discount')).toHaveLength(1);
    expect(calls.filter((row) => row[0] === 'update').length).toBeGreaterThan(0);

    calls.length = 0;
    const second = planPreparationReplay(invoice, {
      meta: { global_discount: { percent: 10 } },
      lines: [
        { id: 1, reservation_id: 11, description: clinicLines[0].description, line_total: 70 },
        { id: 2, reservation_id: 12, description: clinicLines[1].description, line_total: 95 },
      ],
    });
    expect(second.steps).toEqual([]);
  });

  test('un aller-retour modifié reste attaché à la jambe retour', () => {
    const invoice = buildPreparationInvoice({
      payerType: 'clinic',
      periodYear: 2026,
      periodMonth: 10,
      recipientName: 'Clinique',
      previewLines: [
        {
          booking_id: 11,
          booking_ids: [11, 12],
          is_round_trip_leg: true,
          round_trip_partner_booking_id: 12,
          round_trip_primary_amount_ht: 72,
          round_trip_partner_amount_ht: 72,
          description: 'Aller Anieres → clinique',
          round_trip_partner_description: 'Retour clinique → Anieres',
          amount_ht: 144,
          scheduled_at: '2026-10-06T09:10:00',
          patient_name: 'Claire Bonvin',
        },
      ],
    });
    expect(invoice.lines).toHaveLength(2);
    expect(invoice.lines[0].match_ids).toEqual(['11']);
    expect(invoice.lines[1].match_ids).toEqual(['12']);
    expect(invoice.total_amount).toBe(144);
    const edited = updatePreparationLine(invoice, invoice.lines[1].id, {
      description: 'Retour modifié',
      line_total: 60,
    });
    const plan = planPreparationReplay(edited, {
      lines: [
        {
          id: 501,
          reservation_id: 11,
          description: 'Aller Anieres → clinique',
          line_total: 72,
        },
        {
          id: 502,
          reservation_id: 12,
          description: 'Retour clinique → Anieres',
          line_total: 72,
        },
      ],
    });
    expect(plan.steps).toEqual([
      {
        type: 'update',
        lineId: 502,
        body: { description: 'Retour modifié', line_total: 60 },
      },
    ]);
  });

  test('la relecture serveur a lieu même sans modification locale', async () => {
    const invoice = buildClinic();
    const server = {
      id: 90,
      invoice_number: 'EM-2026-10-0008',
      lines: invoice.lines.map((line, index) => ({
        id: 500 + index,
        reservation_id: line.reservation_id,
        description: line.description,
        line_total: line.preparation_base,
      })),
    };
    const load = jest.fn(async () => server);
    const opened = await applyPreparationToPersistedInvoice(
      { load, async updateLine() {}, async applyGlobalDiscount() {}, async removeLine() {}, async addCustomLine() {} },
      server,
      invoice
    );
    expect(load).toHaveBeenCalledTimes(1);
    expect(opened.invoice_number).toBe('EM-2026-10-0008');
    expect(planPreparationReplay(invoice, server).steps).toEqual([]);
  });

  test('patient, institution et partenaire produisent le même shell', () => {
    const line = { booking_id: 1, description: 'Trajet', amount_ht: 40, scheduled_at: '2026-10-01' };
    const patient = buildPreparationInvoice({
      payerType: 'patient',
      periodYear: 2026,
      periodMonth: 10,
      recipientName: 'Ada Lovelace',
      previewLines: [line],
    });
    const partner = buildPreparationInvoice({
      payerType: 'partner',
      periodYear: 2026,
      periodMonth: 10,
      recipientName: 'MobilCar',
      previewLines: [{ ...line, pickup_label: '08:23' }],
    });
    expect(patient.client.patient_display_name).toBe('Ada Lovelace');
    expect(patient.invoice_number).toBe('');
    expect(partner.billing_strategy).toBe('partner_monthly');
    expect(partner.line_time_mode).toBe('none');
    const partnerWithoutClock = buildPreparationInvoice({
      payerType: 'partner',
      periodYear: 2026,
      periodMonth: 10,
      recipientName: 'MobilCar',
      previewLines: [line],
    });
    expect(partnerWithoutClock.line_time_mode).toBe('none');
    expect(partner.editor_header).toBe(true);
    expect(partner.invoice_number).toBe('');
    const optedIn = { ...partner, line_time_mode: 'pickup' };
    expect(
      planPreparationReplay(optedIn, { line_time_mode: 'none', lines: [] }).steps
    ).toEqual([{ type: 'header', body: { line_time_mode: 'pickup' } }]);
    expect(planPreparationReplay(partner, { line_time_mode: 'none', lines: [] }).steps).toEqual(
      []
    );
  });
});
