import {
  formatNameWithCivility,
  getGenderLabel,
  getGenderShortLabel,
  resolvePassengerGender,
} from '../personCivility';

describe('personCivility', () => {
  it('mappe les genres connus', () => {
    expect(getGenderShortLabel('FEMME')).toBe('Mme');
    expect(getGenderShortLabel('HOMME')).toBe('M.');
    expect(getGenderLabel('FEMME')).toBe('Madame');
    expect(getGenderLabel('HOMME')).toBe('Monsieur');
  });

  it('ignore AUTRE', () => {
    expect(getGenderShortLabel('AUTRE')).toBeNull();
    expect(getGenderLabel('AUTRE')).toBe('Autre');
  });

  it('préfixe le nom', () => {
    expect(formatNameWithCivility('Matsa CHERIF', 'FEMME')).toBe('Mme Matsa CHERIF');
    expect(formatNameWithCivility('Jean Dupont', null)).toBe('Jean Dupont');
  });

  it('résout le genre depuis identity puis passenger puis client', () => {
    expect(resolvePassengerGender(
      { passenger: { gender: 'HOMME' }, client: { gender: 'FEMME' } },
      { passenger: { gender: 'FEMME' } },
    )).toBe('FEMME');
    expect(resolvePassengerGender(
      { passenger: { gender: 'HOMME' } },
      null,
    )).toBe('HOMME');
    expect(resolvePassengerGender(
      { client: { gender: 'FEMME' } },
      null,
    )).toBe('FEMME');
  });
});
