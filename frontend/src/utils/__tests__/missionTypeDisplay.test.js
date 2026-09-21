import {
  formatDeliveryDescriptionDisplay,
  formatMissionTypeLabel,
  getDeliveryDescription,
  isMaterialDelivery,
  MISSING_DELIVERY_DESCRIPTION,
  MISSION_DELIVERY_BADGE,
  normalizeMissionType,
} from '../missionTypeDisplay';

describe('missionTypeDisplay', () => {
  it('normalise un type vide en transport patient', () => {
    expect(normalizeMissionType(null)).toBe('patient_transport');
    expect(normalizeMissionType('')).toBe('patient_transport');
    expect(normalizeMissionType('Material_Delivery')).toBe('material_delivery');
  });

  it('libellé FR pour les types connus', () => {
    expect(formatMissionTypeLabel('material_delivery')).toBe('Livraison');
    expect(MISSION_DELIVERY_BADGE).toBe('LIVRAISON');
    expect(formatMissionTypeLabel('patient_transport')).toBe('Transport patient');
    expect(formatMissionTypeLabel(null)).toBe('Transport patient');
  });

  it('détecte une livraison sur réservation, offre ou demande', () => {
    expect(isMaterialDelivery({ mission_type: 'material_delivery' })).toBe(true);
    expect(isMaterialDelivery({
      __offer: { transport_request: { mission_type: 'material_delivery' } },
    })).toBe(true);
    expect(isMaterialDelivery({ mission_type: 'patient_transport' })).toBe(false);
    expect(isMaterialDelivery(null)).toBe(false);
  });

  it('extrait la description depuis plusieurs formes API', () => {
    expect(getDeliveryDescription({ delivery_description: '  Oxygène  ' })).toBe('Oxygène');
    expect(getDeliveryDescription({
      transport_request: { delivery_description: 'Dossiers' },
    })).toBe('Dossiers');
    expect(getDeliveryDescription({ mission_type: 'material_delivery' })).toBe('');
    expect(formatDeliveryDescriptionDisplay({
      mission_type: 'material_delivery',
    })).toBe(MISSING_DELIVERY_DESCRIPTION);
  });
});
