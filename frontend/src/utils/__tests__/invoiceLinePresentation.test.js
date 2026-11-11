import {
  invoiceGrandTotalLabel,
  lineDiscountCaption,
  linePreviewAmount,
  pickupDisplayLabel,
  presentPrestationParts,
} from '../invoiceLinePresentation';

const formatClient = (name) => {
  if (name === 'Claire Bonvin') return 'BONVIN Claire';
  if (name === 'Anne Favre') return 'FAVRE Anne';
  return name;
};

test('institution et partenaire partagent Client, Trajet et le total', () => {
  const institution = presentPrestationParts(
    {
      type: 'ride',
      description: "Trajet : Chemin des Vignes 4 → Clinique",
      line_meta: { patient_name: 'Claire Bonvin' },
    },
    { formatClientName: formatClient }
  );
  const partner = presentPrestationParts(
    {
      type: 'ride',
      description: 'Transfert partenaire — Chemin des Vignes 4 → Hôpital de La Tour',
      client_name: 'Anne Favre',
    },
    { showPickup: true, formatClientName: formatClient }
  );
  expect(institution.clientLine).toBe('Client : BONVIN Claire');
  expect(institution.trajetLine).toBe('Trajet : Chemin des Vignes 4 → Clinique');
  expect(institution.pickupLine).toBeNull();
  expect(partner.clientLine).toBe('Client : FAVRE Anne');
  expect(partner.trajetLine).toBe('Trajet : Chemin des Vignes 4 → Hôpital de La Tour');
  expect(partner.pickupLine).toBe('Prise en charge : —');
  expect(partner.trajetLine).not.toMatch(/Transfert partenaire/);
  expect(invoiceGrandTotalLabel('s2_clinic_monthly')).toBe(
    invoiceGrandTotalLabel('partner_monthly')
  );
  expect(invoiceGrandTotalLabel('direct_patient')).toBe(
    invoiceGrandTotalLabel('s2_clinic_monthly')
  );
  expect(invoiceGrandTotalLabel('s1_patient')).toBe('TOTAL À FACTURER :');
});

test('la prise en charge réelle prime, sinon prévue, sinon tiret', () => {
  expect(
    pickupDisplayLabel({
      source: 'actual_boarded',
      scheduled: '13:30',
      boarded: '08:23',
    })
  ).toBe('08:23');
  expect(
    pickupDisplayLabel({
      source: 'scheduled_fallback',
      scheduled: '13:30',
      boarded: '',
    })
  ).toBe('13:30 (prévue)');
  expect(
    pickupDisplayLabel({
      source: 'actual_boarded',
      scheduled: '13:30',
      boarded: '00:00',
    })
  ).toBe('13:30 (prévue)');
  expect(
    pickupDisplayLabel({ source: 'unknown', scheduled: '00:00', boarded: '00:00' })
  ).toBe('—');
});

test('un aller-retour reste deux prestations avec des heures distinctes', () => {
  const outbound = presentPrestationParts(
    {
      type: 'ride',
      client_name: 'Anne Favre',
      departure: 'Chemin des Vignes 4',
      arrival: 'Hôpital de La Tour',
      pickup_time_source: 'actual_boarded',
      boarded_at: '08:23',
      scheduled_pickup_at: '08:00',
    },
    { showPickup: true, formatClientName: formatClient }
  );
  const inbound = presentPrestationParts(
    {
      type: 'ride',
      client_name: 'Anne Favre',
      departure: 'Hôpital de La Tour',
      arrival: 'Chemin des Vignes 4',
      pickup_time_source: 'scheduled_fallback',
      scheduled_pickup_at: '15:40',
    },
    { showPickup: true, formatClientName: formatClient }
  );
  expect(outbound.pickupLine).toBe('Prise en charge : 08:23');
  expect(outbound.trajetLine).toBe('Trajet : Chemin des Vignes 4 → Hôpital de La Tour');
  expect(inbound.pickupLine).toBe('Prise en charge : 15:40 (prévue)');
  expect(inbound.trajetLine).toBe('Trajet : Hôpital de La Tour → Chemin des Vignes 4');
  expect(outbound.clientLine).toBe(inbound.clientLine);
});

test('le rabais de ligne nomme le pourcentage, le prix initial et la déduction', () => {
  const line = {
    id: 7,
    line_total: 64.8,
    line_meta: { original_line_total: 72 },
  };
  const expected = 'Rabais 10 % · prix initial 72.00 CHF · -7.20 CHF';
  expect(
    lineDiscountCaption(line, { meta: { global_discount: { percent: 10, amount_ht: 7.2 } } })
  ).toBeNull();
  expect(
    lineDiscountCaption(line, {
      meta: { per_line_discounts: { lines: [{ line_id: 7, percent: 10 }] } },
    })
  ).toBe(expected);
  expect(lineDiscountCaption({ line_total: 72, line_meta: {} }, {})).toBeNull();
  expect(
    linePreviewAmount(line, {
      meta: { global_discount: { percent: 10, amount_ht: 7.2, subtotal_before_ht: 72 } },
    })
  ).toBe(72);
  expect(linePreviewAmount(line, {})).toBe(64.8);
});
