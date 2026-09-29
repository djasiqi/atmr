import {
  fullNameFromUser,
  looksLikeAccountHandle,
  pickAccountDisplayName,
} from './accountDisplayName';

describe('nom affiché du compte', () => {
  test('prend le prénom et le nom en majuscules', () => {
    expect(fullNameFromUser({ first_name: 'Mirjete', last_name: 'Osmani' })).toBe('Mirjete OSMANI');
  });

  test('ignore l’identifiant technique', () => {
    expect(looksLikeAccountHandle('osmani_mirjete_c8d0da')).toBe(true);
    expect(
      pickAccountDisplayName('osmani_mirjete_c8d0da', 'Mirjete OSMANI')
    ).toBe('Mirjete OSMANI');
    expect(pickAccountDisplayName('osmani_mirjete_c8d0da', '')).toBe('');
  });
});
