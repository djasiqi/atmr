import {
  buildDeliveryPresentation,
  DELIVERY_BENEFICIARY_LABEL,
  formatDeliveryDescriptionDisplay,
  formatMissionTypeLabel,
  isMaterialDelivery,
  MISSING_DELIVERY_DESCRIPTION,
  MISSION_DELIVERY_BADGE,
} from '../missionTypeDisplay';

const FIXTURE_39869 = {
  id: 39869,
  mission_type: 'material_delivery',
  delivery_description: 'Livraison des effets personnels de M. Basset.',
  patient: { first_name: 'Michel', last_name: 'BASSET' },
};

describe('surfaces livraison — fixture #39869', () => {
  it('Institution / Entreprise / Chauffeur partagent le même contrat', () => {
    const presentation = buildDeliveryPresentation(FIXTURE_39869);
    expect(isMaterialDelivery(FIXTURE_39869)).toBe(true);
    expect(presentation.badge).toBe(MISSION_DELIVERY_BADGE);
    expect(presentation.typeLabel).toBe('Livraison');
    expect(presentation.description).toBe('Livraison des effets personnels de M. Basset.');
    expect(presentation.beneficiaryLabel).toBe(DELIVERY_BENEFICIARY_LABEL);
    expect(presentation.beneficiary).toBe('Michel BASSET');
    expect(presentation.cargoLabel).toBe('À transporter');
    expect(formatMissionTypeLabel(FIXTURE_39869.mission_type)).toBe('Livraison');
  });

  it('ne présente pas uniquement le passager comme signal principal', () => {
    const presentation = buildDeliveryPresentation(FIXTURE_39869);
    expect(presentation.badge).toBe('LIVRAISON');
    expect(presentation.beneficiaryLabel).not.toBe('Passager');
  });

  it('n’invente pas une description legacy', () => {
    expect(formatDeliveryDescriptionDisplay({
      mission_type: 'material_delivery',
      delivery_description: null,
    })).toBe(MISSING_DELIVERY_DESCRIPTION);
  });

  it('ne déduit jamais une livraison hors mission_type', () => {
    expect(isMaterialDelivery({
      delivery_description: 'Livraison des effets personnels de M. Basset.',
      patient: { first_name: 'Michel', last_name: 'BASSET' },
    })).toBe(false);
    expect(buildDeliveryPresentation({
      mission_type: 'patient_transport',
      delivery_description: 'Livraison des effets personnels de M. Basset.',
    })).toBeNull();
  });

  it('ignore une delivery_description accidentelle sur un transport de personne', () => {
    const accidental = {
      mission_type: 'patient_transport',
      delivery_description: 'Livraison de documents',
      patient: { first_name: 'Michel', last_name: 'BASSET' },
    };
    expect(isMaterialDelivery(accidental)).toBe(false);
    expect(buildDeliveryPresentation(accidental)).toBeNull();
  });
});
