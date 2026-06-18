import { formatPatientDisplayNameNomPrenom } from '../patientDisplayName';

describe('formatPatientDisplayNameNomPrenom', () => {
  it('normalise Prénom(s) NOM vers NOM Prénom(s)', () => {
    expect(formatPatientDisplayNameNomPrenom('Eliane Francine STOFER-THOMI')).toBe(
      'STOFER-THOMI Eliane Francine'
    );
    expect(formatPatientDisplayNameNomPrenom('Khalid ALAOUI')).toBe('ALAOUI Khalid');
  });

  it('conserve NOM Prénom déjà correct', () => {
    expect(formatPatientDisplayNameNomPrenom('ALEXANDRE Pierre')).toBe('ALEXANDRE Pierre');
    expect(formatPatientDisplayNameNomPrenom('BENDER-BITTAR Chantal-marie')).toBe(
      'BENDER-BITTAR Chantal-Marie'
    );
  });
});
