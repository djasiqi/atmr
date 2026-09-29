import { computeProfileCompletionPercent } from './profileCompletion';

const complete = {
  first_name: 'Mirjete',
  last_name: 'OSMANI',
  email: 'mirjete@example.ch',
  phone: '+41791234567',
  phone_verified: true,
  birth_date: '1980-01-01',
  gender: 'female',
  address: 'Avenue Ernest-Pictet 9, 1203 Genève',
  floor: '3ème',
};

describe('computeProfileCompletionPercent', () => {
  test('un numéro non confirmé empêche les 100 %', () => {
    expect(computeProfileCompletionPercent({ ...complete, phone_verified: false })).toBe(85);
    expect(computeProfileCompletionPercent({ ...complete, phone_verified: true })).toBe(100);
  });

  test('un numéro vide ne compte pas comme vérifié', () => {
    expect(computeProfileCompletionPercent({ ...complete, phone: '', phone_verified: true })).toBe(85);
  });
});
