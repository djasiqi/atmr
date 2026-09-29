import {
  classifyPortalMedicalPlace,
  composeProfilePickupAccess,
  destinationLooksMedical,
  PORTAL_DOCTOR_FACILITY_LABEL,
  portalTransportProfileGaps,
} from '../../pages/client/Dashboard/portalTransportProfile';

describe('Profil prêt au transport', () => {
  const ready = {
    first_name: 'Ada',
    last_name: 'Martin',
    birth_date: '1980-01-01',
    phone: '+41791234567',
    phone_verified: true,
    domicile: { address: 'Rue du Test 1', lat: 46.2, lon: 6.14 },
    access: { floor: '3e', door_code: 'Osmani', notes: "Accès par l'entrée arrière" },
  };

  it('exige le téléphone vérifié et une adresse choisie', () => {
    expect(portalTransportProfileGaps(ready)).toEqual([]);
    expect(portalTransportProfileGaps({ ...ready, phone_verified: false })).toContain('phone');
    expect(
      portalTransportProfileGaps({
        ...ready,
        domicile: { address: 'Rue du Test 1', lat: null, lon: null },
      })
    ).toContain('address_validated');
  });

  it('compose l’accès habituel sans réécrire une course', () => {
    expect(composeProfilePickupAccess(ready)).toBe(
      "3e étage\nInterphone Osmani\nAccès par l'entrée arrière"
    );
  });

  it('reconnaît un lieu de soins et ignore une adresse ordinaire', () => {
    expect(destinationLooksMedical('Hôpital de la Tour')).toBe(true);
    expect(destinationLooksMedical('Clinique de Carouge')).toBe(true);
    expect(destinationLooksMedical('Cabinet médical d’ophtalmologie')).toBe(true);
    expect(destinationLooksMedical('Chirurgie dentaire, Dr Martin')).toBe(true);
    expect(destinationLooksMedical('Dentiste des Eaux-Vives')).toBe(true);
    expect(destinationLooksMedical('HUG, rue Gabrielle-Perret-Gentil')).toBe(true);
    expect(destinationLooksMedical('Gare de Lausanne')).toBe(false);
    expect(destinationLooksMedical('Un endroit calme')).toBe(false);
    expect(destinationLooksMedical('Orléans, centre-ville')).toBe(false);
  });

  it('range un médecin dans Médecin et propose un cabinet médical', () => {
    expect(classifyPortalMedicalPlace('Dr méd. Bigler Jean-Michel')).toEqual({
      facility: PORTAL_DOCTOR_FACILITY_LABEL,
      doctor: 'Dr méd. Bigler Jean-Michel',
    });
    expect(classifyPortalMedicalPlace('Clinique de Carouge, rue du Marché')).toEqual({
      facility: 'Clinique de Carouge',
      doctor: '',
    });
    expect(
      classifyPortalMedicalPlace('Clinique de Joli-Mont, Avenue Trembley 45, 1209, Genève')
    ).toEqual({
      facility: 'Clinique de Joli-Mont',
      doctor: '',
    });
  });
});
