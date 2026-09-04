import {
  adultDobCutoff,
  buildInstitutionPatientPayload,
  isMinorDob,
  normalizeDob,
  normalizeGender,
  normalizeEmail,
  patientAgeYears,
  sanitizeAvsInput,
  sanitizePhoneInput,
  sanitizePostalCodeInput,
  todayIso,
} from './institutionPatientForm';

function isoYearsAgo(years, dayOffset = 0) {
  const t = new Date();
  const d = new Date(t.getFullYear() - years, t.getMonth(), t.getDate() + dayOffset);
  const pad = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}

const COMPLETE_DOMICILE = {
  address: '12 rue du Lac',
  postal_code: '1200',
  city: 'Genève',
};

describe('institutionPatientForm — PATIENT-IDENTITY-01', () => {
  it('sanitizePhoneInput removes spaces', () => {
    expect(sanitizePhoneInput('+41 76 744 72 72')).toBe('+41767447272');
  });

  it('sanitizePostalCodeInput keeps digits only', () => {
    expect(sanitizePostalCodeInput('12 47')).toBe('1247');
  });

  it('sanitizeAvsInput keeps digits and dots', () => {
    expect(sanitizeAvsInput('756 1234.5678.97')).toBe('7561234.5678.97');
  });

  it('normalizeEmail lowercases and trims', () => {
    expect(normalizeEmail('  Curateur@Example.CH ')).toEqual({
      value: 'curateur@example.ch',
      error: null,
    });
  });

  it('civilité absente → erreur', () => {
    expect(normalizeGender('')).toEqual({ value: null, error: 'Civilité requise' });
  });

  it('DOB absente → erreur', () => {
    expect(normalizeDob('').error).toBe('Date de naissance requise');
  });

  it('DOB future → refus', () => {
    expect(normalizeDob('2027-01-01').error).toMatch(/futur/i);
  });

  it('aujourd’hui → mineur accepté (avec confirmation)', () => {
    const today = todayIso();
    const r = normalizeDob(today);
    expect(r.error).toBeNull();
    expect(r.isMinor).toBe(true);
    expect(r.age).toBe(0);
  });

  it('exactement 18 ans → adulte normal', () => {
    const iso = adultDobCutoff();
    const r = normalizeDob(iso);
    expect(r.error).toBeNull();
    expect(r.isMinor).toBe(false);
    expect(isMinorDob(iso)).toBe(false);
  });

  it('17 ans → mineur accepté', () => {
    const iso = isoYearsAgo(18, 1);
    const r = normalizeDob(iso);
    expect(r.error).toBeNull();
    expect(r.isMinor).toBe(true);
  });

  it('date calendrier invalide → refus', () => {
    expect(normalizeDob('2026-02-31').error).toMatch(/invalide/i);
  });

  it('création mineur sans confirmation → needsMinorConfirmation', () => {
    const { errors, needsMinorConfirmation, payload } = buildInstitutionPatientPayload({
      first_name: 'A',
      last_name: 'B',
      gender: 'HOMME',
      dob: todayIso(),
      ...COMPLETE_DOMICILE,
    });
    expect(errors).toEqual([]);
    expect(needsMinorConfirmation).toBe(true);
    expect(payload.minor_dob_confirmed).toBeUndefined();
  });

  it('création mineur confirmée → payload.minor_dob_confirmed', () => {
    const { payload, needsMinorConfirmation } = buildInstitutionPatientPayload(
      {
        first_name: 'A',
        last_name: 'B',
        gender: 'HOMME',
        dob: todayIso(),
        ...COMPLETE_DOMICILE,
      },
      { minorDobConfirmed: true },
    );
    expect(needsMinorConfirmation).toBe(true);
    expect(payload.minor_dob_confirmed).toBe(true);
  });

  it('édition mineur DOB inchangée → pas de reconfirmation', () => {
    const minorIso = todayIso();
    const { needsMinorConfirmation } = buildInstitutionPatientPayload(
      {
        first_name: 'A',
        last_name: 'B',
        gender: 'HOMME',
        dob: minorIso,
        phone: '+41791234567',
        ...COMPLETE_DOMICILE,
      },
      { previousDob: minorIso },
    );
    expect(needsMinorConfirmation).toBe(false);
  });
  it('buildInstitutionPatientPayload accepte un adulte complet', () => {
    const { payload, errors, needsMinorConfirmation } = buildInstitutionPatientPayload({
      first_name: ' Jean ',
      last_name: ' dupont ',
      gender: 'HOMME',
      dob: '1985-03-15',
      phone: '+41 79 123 45 67',
      address: '12 rue du Lac',
      postal_code: '12 04',
      city: 'Genève',
      has_guardianship: false,
    });

    expect(errors).toEqual([]);
    expect(needsMinorConfirmation).toBe(false);
    expect(payload.first_name).toBe('Jean');
    expect(payload.last_name).toBe('DUPONT');
    expect(payload.gender).toBe('HOMME');
    expect(payload.dob).toBe('1985-03-15');
    expect(payload.address).toBe('12 rue du Lac');
    expect(payload.postal_code).toBe('1204');
    expect(payload.city).toBe('Genève');
  });

  it('création sans adresse → refus', () => {
    const { errors } = buildInstitutionPatientPayload({
      first_name: 'A',
      last_name: 'B',
      gender: 'HOMME',
      dob: '1985-03-15',
      postal_code: '1200',
      city: 'Genève',
    });
    expect(errors).toContain('Adresse requise');
  });

  it('création sans NPA → refus', () => {
    const { errors } = buildInstitutionPatientPayload({
      first_name: 'A',
      last_name: 'B',
      gender: 'HOMME',
      dob: '1985-03-15',
      address: '12 rue',
      city: 'Genève',
    });
    expect(errors).toContain('NPA requis');
  });

  it('création sans ville → refus', () => {
    const { errors } = buildInstitutionPatientPayload({
      first_name: 'A',
      last_name: 'B',
      gender: 'HOMME',
      dob: '1985-03-15',
      address: '12 rue',
      postal_code: '1200',
    });
    expect(errors).toContain('Ville requise');
  });

  it('création domicile blanc → refus', () => {
    const { errors } = buildInstitutionPatientPayload({
      first_name: 'A',
      last_name: 'B',
      gender: 'HOMME',
      dob: '1985-03-15',
      address: '   ',
      postal_code: '  ',
      city: '\t',
    });
    expect(errors).toEqual(expect.arrayContaining([
      'Adresse requise',
      'NPA requis',
      'Ville requise',
    ]));
  });

  it('force_create + mineur sans confirmation → needsMinorConfirmation (pas de bypass)', () => {
    const { needsMinorConfirmation, payload, errors } = buildInstitutionPatientPayload(
      {
        first_name: 'Julie',
        last_name: 'Dupont',
        gender: 'FEMME',
        dob: todayIso(),
        ...COMPLETE_DOMICILE,
      },
      { forceCreate: true },
    );
    expect(errors).toEqual([]);
    expect(needsMinorConfirmation).toBe(true);
    expect(payload.force_create).toBe(true);
    expect(payload.minor_dob_confirmed).toBeUndefined();
  });

  it('force_create + mineur confirmé → les deux flags dans le payload', () => {
    const { payload, needsMinorConfirmation } = buildInstitutionPatientPayload(
      {
        first_name: 'Julie',
        last_name: 'Dupont',
        gender: 'FEMME',
        dob: todayIso(),
        ...COMPLETE_DOMICILE,
      },
      { forceCreate: true, minorDobConfirmed: true },
    );
    expect(needsMinorConfirmation).toBe(true);
    expect(payload.force_create).toBe(true);
    expect(payload.minor_dob_confirmed).toBe(true);
  });

  it('adulte → minor_dob_confirmed ignoré côté payload (pas de flag)', () => {
    const { payload, needsMinorConfirmation } = buildInstitutionPatientPayload(
      {
        first_name: 'Adulte',
        last_name: 'Flag',
        gender: 'HOMME',
        dob: '1985-03-15',
        ...COMPLETE_DOMICILE,
      },
      { minorDobConfirmed: true },
    );
    expect(needsMinorConfirmation).toBe(false);
    expect(payload.minor_dob_confirmed).toBeUndefined();
  });

  it('mineur → autre DOB mineure → nouvelle confirmation', () => {
    const prev = isoYearsAgo(16);
    const next = isoYearsAgo(15);
    const { needsMinorConfirmation } = buildInstitutionPatientPayload(
      {
        first_name: 'A',
        last_name: 'B',
        gender: 'FEMME',
        dob: next,
        ...COMPLETE_DOMICILE,
      },
      { previousDob: prev },
    );
    expect(needsMinorConfirmation).toBe(true);
  });

  it('édition téléphone seulement (DOB mineure inchangée) → pas de reconfirmation', () => {
    const minorIso = isoYearsAgo(16);
    const { needsMinorConfirmation, errors } = buildInstitutionPatientPayload(
      {
        first_name: 'Julie',
        last_name: 'Dupont',
        gender: 'FEMME',
        dob: minorIso,
        phone: '+41791112233',
        ...COMPLETE_DOMICILE,
      },
      { previousDob: minorIso },
    );
    expect(errors).toEqual([]);
    expect(needsMinorConfirmation).toBe(false);
  });

  it('édition civilité seulement (DOB mineure inchangée) → pas de reconfirmation', () => {
    const minorIso = isoYearsAgo(16);
    const { needsMinorConfirmation } = buildInstitutionPatientPayload(
      {
        first_name: 'Julie',
        last_name: 'Dupont',
        gender: 'HOMME',
        dob: minorIso,
        ...COMPLETE_DOMICILE,
      },
      { previousDob: minorIso },
    );
    expect(needsMinorConfirmation).toBe(false);
  });

  it('legacy sans DOB → régularisation exigée à l’édition complète', () => {
    const { errors } = buildInstitutionPatientPayload({
      first_name: 'Legacy',
      last_name: 'SansDob',
      gender: 'FEMME',
      dob: '',
      ...COMPLETE_DOMICILE,
    });
    expect(errors.some((e) => /naissance/i.test(e))).toBe(true);
  });

  it('legacy sans adresse → régularisation exigée à l’édition complète', () => {
    const { errors } = buildInstitutionPatientPayload({
      first_name: 'Legacy',
      last_name: 'SansAddr',
      gender: 'FEMME',
      dob: '1985-03-15',
    });
    expect(errors).toEqual(expect.arrayContaining([
      'Adresse requise',
      'NPA requis',
      'Ville requise',
    ]));
  });

  it('29.02 année bissextile — âge avant / après anniversaire', () => {
    const dob = '2008-02-29';
    expect(patientAgeYears(dob, new Date(2026, 1, 28))).toBe(17);
    expect(patientAgeYears(dob, new Date(2026, 2, 1))).toBe(18);
    expect(isMinorDob(dob, new Date(2026, 1, 28))).toBe(true);
    expect(isMinorDob(dob, new Date(2026, 2, 1))).toBe(false);
  });
});
