/**
 * Tests pour la recherche client (filtre global page Gestion des clients).
 * Vérifie que first_name, last_name, full_name, username, adresse sont bien matchés.
 */

import {
  getClientDisplayName,
  normalizeText,
  buildSearchHaystack,
  clientMatchesSearch,
  matchSearchQuery,
} from './clientSearchUtils';

describe('clientSearchUtils', () => {
  describe('getClientDisplayName', () => {
    it('institution → institution_name', () => {
      expect(getClientDisplayName({
        id: 1,
        is_institution: true,
        institution_name: 'Clinique X',
      })).toBe('Clinique X');
    });

    it('full_name si présent et ≠ "Nom non renseigné"', () => {
      expect(getClientDisplayName({
        id: 1,
        is_institution: false,
        full_name: 'Anna-Rosa El-Alaoui',
      })).toBe('Anna-Rosa El-Alaoui');
      expect(getClientDisplayName({
        id: 1,
        full_name: 'Nom non renseigné',
        first_name: 'Jean',
        last_name: 'Dupont',
      })).toBe('Jean Dupont');
    });

    it('sinon first_name + last_name (fallback user_*)', () => {
      expect(getClientDisplayName({
        id: 1,
        first_name: 'Jean',
        last_name: 'Dupont',
      })).toBe('Jean Dupont');
      expect(getClientDisplayName({
        id: 42,
        user_first_name: 'Marie',
        user_last_name: 'Martin',
      })).toBe('Marie Martin');
    });

    it('fallback Client #id si aucun nom', () => {
      expect(getClientDisplayName({ id: 99 })).toBe('Client #99');
      expect(getClientDisplayName({ id: 1, full_name: 'Nom non renseigné' })).toBe('Client #1');
    });

    it('null/undefined → ""', () => {
      expect(getClientDisplayName(null)).toBe('');
      expect(getClientDisplayName(undefined)).toBe('');
    });
  });

  describe('normalizeText', () => {
    it('lowercase et enlève les accents', () => {
      expect(normalizeText('Élève')).toBe('eleve');
      expect(normalizeText('  François  ')).toBe('francois');
    });

    it('tirets et apostrophes → espace, puis collapse', () => {
      expect(normalizeText('El-Alaoui')).toBe('el alaoui');
      expect(normalizeText("D'Artagnan")).toBe('d artagnan');
      expect(normalizeText("d'artagnan")).toBe('d artagnan');
      expect(normalizeText('Jean  Pierre')).toBe('jean pierre');
      expect(normalizeText('a   b   c')).toBe('a b c');
    });

    it('gère null/undefined', () => {
      expect(normalizeText(null)).toBe('');
      expect(normalizeText(undefined)).toBe('');
    });
  });

  describe('buildSearchHaystack', () => {
    it('concatène first_name, last_name, full_name, id, email, domicile', () => {
      const client = {
        id: 42,
        first_name: 'Anna-Rosa',
        last_name: 'El-Alaoui',
        full_name: 'Anna-Rosa El-Alaoui',
        contact_email: 'anna@example.ch',
        domicile: { address: 'Rue X 1', zip: '1200', city: 'Genève' },
      };
      const h = buildSearchHaystack(client);
      expect(h).toContain('42');
      expect(h).toContain('Anna-Rosa');
      expect(h).toContain('El-Alaoui');
      expect(h).toContain('Anna-Rosa El-Alaoui');
      expect(h).toContain('anna@example.ch');
      expect(h).toContain('Rue X 1');
      expect(h).toContain('1200');
      expect(h).toContain('Genève');
    });

    it('utilise user_first_name / user_last_name si first_name / last_name absents', () => {
      const client = {
        id: 1,
        user_first_name: 'Jean',
        user_last_name: 'Dupont',
      };
      const h = buildSearchHaystack(client);
      expect(h).toContain('Jean');
      expect(h).toContain('Dupont');
    });

    it('inclut username (user) et institution_name', () => {
      const client = {
        id: 1,
        first_name: 'A',
        last_name: 'B',
        user: { username: 'a.b' },
        institution_name: 'Clinique X',
      };
      const h = buildSearchHaystack(client);
      expect(h).toContain('a.b');
      expect(h).toContain('Clinique X');
    });

    it('inclut birth_date en YYYY-MM-DD, dd/mm/yyyy, dd.mm.yyyy, ddmmyyyy', () => {
      const client = {
        id: 1,
        first_name: 'Jean',
        last_name: 'Dupont',
        user: { birth_date: '1990-05-15' },
      };
      const h = buildSearchHaystack(client);
      expect(h).toContain('1990-05-15');
      expect(h).toContain('15/05/1990');
      expect(h).toContain('15.05.1990');
      expect(h).toContain('15051990');
    });
  });

  describe('clientMatchesSearch', () => {
    it('"anna" matche "Anna-Rosa El-Alaoui"', () => {
      const client = {
        id: 1,
        first_name: 'Anna-Rosa',
        last_name: 'El-Alaoui',
        full_name: 'Anna-Rosa El-Alaoui',
      };
      expect(clientMatchesSearch(client, 'anna')).toBe(true);
      expect(clientMatchesSearch(client, 'ANNA')).toBe(true);
    });

    it('"el-alaoui" et "el alaoui" matchent "Anna-Rosa El-Alaoui" (tiret ↔ espace)', () => {
      const client = {
        id: 1,
        first_name: 'Anna-Rosa',
        last_name: 'El-Alaoui',
        full_name: 'Anna-Rosa El-Alaoui',
      };
      expect(clientMatchesSearch(client, 'el-alaoui')).toBe(true);
      expect(clientMatchesSearch(client, 'el alaoui')).toBe(true);
      expect(clientMatchesSearch(client, 'El-Alaoui')).toBe(true);
    });

    it('"d\'artagnan" / "d artagnan" matchent "D\'Artagnan" (apostrophe → espace)', () => {
      const client = {
        id: 1,
        first_name: "D'Artagnan",
        last_name: 'Dupont',
        full_name: "D'Artagnan Dupont",
      };
      expect(clientMatchesSearch(client, "d'artagnan")).toBe(true);
      expect(clientMatchesSearch(client, 'd artagnan')).toBe(true);
    });

    it('"annemasse" matche via adresse (ville)', () => {
      const client = {
        id: 1,
        first_name: 'Jean',
        last_name: 'Dupont',
        domicile: { address: 'Rue de la Gare 1', zip: '74100', city: 'Annemasse' },
      };
      expect(clientMatchesSearch(client, 'annemasse')).toBe(true);
      expect(clientMatchesSearch(client, 'Annemasse')).toBe(true);
    });

    it('"annemasse" matche via adresse (address)', () => {
      const client = {
        id: 1,
        first_name: 'Marie',
        last_name: 'Martin',
        domicile: { address: '12 chemin d\'Annemasse', zip: '74000', city: 'Ville' },
      };
      expect(clientMatchesSearch(client, 'annemasse')).toBe(true);
    });

    it('query vide matche tout', () => {
      const client = { id: 1, first_name: 'X', last_name: 'Y' };
      expect(clientMatchesSearch(client, '')).toBe(true);
      expect(clientMatchesSearch(client, '   ')).toBe(true);
    });

    it('sans match retourne false', () => {
      const client = {
        id: 1,
        first_name: 'Anna-Rosa',
        last_name: 'El-Alaoui',
        domicile: { city: 'Genève' },
      };
      expect(clientMatchesSearch(client, 'xyz')).toBe(false);
      expect(clientMatchesSearch(client, 'annemasse')).toBe(false);
    });

    it('match insensible aux accents', () => {
      const client = { id: 1, first_name: 'François', last_name: 'Müller' };
      expect(clientMatchesSearch(client, 'francois')).toBe(true);
      expect(clientMatchesSearch(client, 'muller')).toBe(true);
    });

    it('birth_date : dd.mm.yyyy et ddmmyyyy matchent', () => {
      const client = {
        id: 1,
        first_name: 'Jean',
        last_name: 'Dupont',
        user: { birth_date: '1990-05-15' },
      };
      expect(clientMatchesSearch(client, '15.05.1990')).toBe(true);
      expect(clientMatchesSearch(client, '15051990')).toBe(true);
      expect(clientMatchesSearch(client, '15/05/1990')).toBe(true);
      expect(clientMatchesSearch(client, '1990-05-15')).toBe(true);
    });
  });

  describe('matchSearchQuery (haystack pré-normalisé)', () => {
    it('query vide matche', () => {
      expect(matchSearchQuery('abc def', '')).toBe(true);
      expect(matchSearchQuery('abc def', '   ')).toBe(true);
    });
    it('match lorsque le haystack normalisé contient la query normalisée', () => {
      const h = normalizeText(buildSearchHaystack({ first_name: 'Anna-Rosa', last_name: 'El-Alaoui' }));
      expect(matchSearchQuery(h, 'anna')).toBe(true);
      expect(matchSearchQuery(h, 'el-alaoui')).toBe(true);
      expect(matchSearchQuery(h, 'el alaoui')).toBe(true);
    });
    it('pas de match retourne false', () => {
      expect(matchSearchQuery('abc def', 'xyz')).toBe(false);
    });
  });
});
