/**
 * Tests unitaires pour le helper phone (normalisation et validation).
 * Aligné sur le backend: 7 à 15 chiffres, option '+' (E.164 simplifié).
 */

import {
  normalizePhone,
  isValidPhone,
  getPhoneValidationError,
  PHONE_REGEX,
} from './phone';

describe('normalizePhone', () => {
  it('retourne null pour null/undefined/vide', () => {
    expect(normalizePhone(null)).toBe(null);
    expect(normalizePhone(undefined)).toBe(null);
    expect(normalizePhone('')).toBe(null);
    expect(normalizePhone('   ')).toBe(null);
  });

  it('supprime espaces, tirets, parenthèses, points', () => {
    expect(normalizePhone('079 123 45 67')).toBe('0791234567');
    expect(normalizePhone('079-123-45-67')).toBe('0791234567');
    expect(normalizePhone('+41 (0)79 123 45 67')).toBe('+41791234567');
    expect(normalizePhone('022.123.45.67')).toBe('0221234567');
  });

  it('convertit 00 en +', () => {
    expect(normalizePhone('0041791234567')).toBe('+41791234567');
    expect(normalizePhone('00 41 79 123 45 67')).toBe('+41791234567');
  });

  it('conserve + optionnel en tête', () => {
    expect(normalizePhone('+41791234567')).toBe('+41791234567');
    expect(normalizePhone('41791234567')).toBe('41791234567');
  });

  it('ne garde que chiffres et + en première position', () => {
    expect(normalizePhone('+41 79 123 45 67')).toBe('+41791234567');
    expect(normalizePhone('abc0791234567')).toBe('0791234567');
  });
});

describe('isValidPhone', () => {
  it('accepte null/vide comme valide', () => {
    expect(isValidPhone(null)).toBe(true);
    expect(isValidPhone('')).toBe(true);
  });

  it('accepte 7 à 15 chiffres avec + optionnel', () => {
    expect(isValidPhone('0791234')).toBe(true);
    expect(isValidPhone('0791234567')).toBe(true);
    expect(isValidPhone('+41791234567')).toBe(true);
    expect(isValidPhone('123456789012345')).toBe(true);
  });

  it('rejette moins de 7 chiffres', () => {
    expect(isValidPhone('123456')).toBe(false);
  });

  it('rejette plus de 15 chiffres', () => {
    expect(isValidPhone('1234567890123456')).toBe(false);
  });

  it('rejette lettres ou caractères invalides', () => {
    expect(isValidPhone('079 123 45 67')).toBe(false);
    expect(isValidPhone('079-123-45-67')).toBe(false);
  });
});

describe('getPhoneValidationError', () => {
  it('retourne null pour valeur valide ou vide', () => {
    expect(getPhoneValidationError(null)).toBe(null);
    expect(getPhoneValidationError('')).toBe(null);
    expect(getPhoneValidationError('0791234567')).toBe(null);
    expect(getPhoneValidationError('+41791234567')).toBe(null);
  });

  it('retourne un message pour valeur invalide', () => {
    expect(getPhoneValidationError('abc')).toBeTruthy();
    expect(getPhoneValidationError('123456')).toBeTruthy();
  });
});

describe('PHONE_REGEX', () => {
  it('matche format backend ^+?\\d{7,15}$', () => {
    expect(PHONE_REGEX.test('0791234')).toBe(true);
    expect(PHONE_REGEX.test('0791234567')).toBe(true);
    expect(PHONE_REGEX.test('+41791234567')).toBe(true);
    expect(PHONE_REGEX.test('123456789012345')).toBe(true);
    expect(PHONE_REGEX.test('079 123 45 67')).toBe(false);
    expect(PHONE_REGEX.test('123456')).toBe(false);
    expect(PHONE_REGEX.test('1234567890123456')).toBe(false);
  });
});
