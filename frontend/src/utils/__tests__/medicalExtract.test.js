import {
  extractEstablishmentLabel,
  extractMedicalServiceInfo,
} from '../medicalExtract';

describe('extractEstablishmentLabel', () => {
  it('conserve le nom complet avec acronyme entre parenthèses', () => {
    expect(
      extractEstablishmentLabel(
        'Hôpitaux Universitaires de Genève (HUG), Rue Gabrielle-Perret-Gentil 4, 1205 Genève'
      )
    ).toBe('Hôpitaux Universitaires de Genève (HUG)');
  });

  it('retourne le nom complet s’il n’y a pas d’acronyme', () => {
    expect(extractEstablishmentLabel('Clinique La Colline, Avenue de la Roseraie 76')).toBe(
      'Clinique La Colline'
    );
  });
});

describe('extractMedicalServiceInfo', () => {
  it('HUG : établissement = nom complet, pas de service « Hôpitaux »', () => {
    const extracted = extractMedicalServiceInfo(
      'Hôpitaux Universitaires de Genève (HUG), Rue Gabrielle-Perret-Gentil 4, 1205 Genève'
    );
    expect(extracted.medical_facility).toBe('Hôpitaux Universitaires de Genève (HUG)');
    expect(extracted.hospital_service).toBeUndefined();
  });

  it('conserve un vrai service médical', () => {
    const extracted = extractMedicalServiceInfo('HUG — Service de Cardiologie, Bâtiment A');
    expect(extracted.medical_facility).toBe('HUG — Service de Cardiologie');
    expect(extracted.hospital_service?.toLowerCase()).toContain('cardiologie');
    expect(extracted.building?.toLowerCase()).toContain('bâtiment');
  });

  it('détecte une spécialité seule (Oncologie)', () => {
    const extracted = extractMedicalServiceInfo('Rendez-vous Oncologie étage 2');
    expect(extracted.hospital_service).toBe('Oncologie');
    expect(extracted.floor?.toLowerCase()).toContain('étage');
  });

  it('extrait un médecin', () => {
    const extracted = extractMedicalServiceInfo('Dr Jean Dupont, Cabinet médical, Rue X 1');
    expect(extracted.doctor_name?.toLowerCase()).toContain('dupont');
  });
});
