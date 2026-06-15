import {
  buildInstitutionPatientPayload,
  normalizeEmail,
  sanitizeAvsInput,
  sanitizePhoneInput,
  sanitizePostalCodeInput,
} from './institutionPatientForm';

describe('institutionPatientForm', () => {
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

  it('buildInstitutionPatientPayload normalizes phone and clears guardianship fields', () => {
    const { payload, errors } = buildInstitutionPatientPayload({
      first_name: ' Jean ',
      last_name: ' dupont ',
      phone: '+41 79 123 45 67',
      postal_code: '12 04',
      has_guardianship: false,
      guardian_phone: '+41 22 000 00 00',
      guardian_email: 'x@y.ch',
    });

    expect(errors).toEqual([]);
    expect(payload.first_name).toBe('Jean');
    expect(payload.last_name).toBe('DUPONT');
    expect(payload.phone).toBe('+41791234567');
    expect(payload.postal_code).toBe('1204');
    expect(payload.guardian_phone).toBeNull();
    expect(payload.guardian_email).toBeNull();
  });
});
