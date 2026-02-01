import { shortAddress } from 'pages/company/Dashboard/components/formatAddress';

describe('shortAddress', () => {
  it('retourne raw si déjà court', () => {
    expect(shortAddress('Rue X 12, Genève')).toBe('Rue X 12, Genève');
    expect(shortAddress('Genève')).toBe('Genève');
  });

  it('trim et collapse spaces', () => {
    expect(shortAddress('  Rue  X   12  ,  Genève  ')).toBe('Rue X 12, Genève');
  });

  it('garde la partie avant " - " ou " — "', () => {
    expect(shortAddress('HUG Maternité - Boulevard de la Cluse, 1205 Genève')).toBe('HUG Maternité');
    expect(shortAddress('Nom lieu — Rue longue, 1205 Genève')).toBe('Nom lieu');
  });

  it('prend les 2 premiers segments si plusieurs virgules', () => {
    expect(shortAddress('Rue X 12, 1205 Genève, Suisse')).toBe('Rue X 12, 1205 Genève');
  });

  it('supprime ", Suisse" et ", Switzerland"', () => {
    expect(shortAddress('Rue X 12, Genève, Suisse')).toBe('Rue X 12, Genève');
    expect(shortAddress('Rue X 12, Genève, Switzerland')).toBe('Rue X 12, Genève');
  });

  it('tronque à 48 chars avec ellipsis si trop long', () => {
    const long =
      'Rue Gabrielle-Perret-Gentil 4, Bâtiment principal niveau 2, 1205 Genève, Suisse';
    const result = shortAddress(long);
    expect(result.length).toBeLessThanOrEqual(49);
    expect(result.endsWith('…')).toBe(true);
  });

  it('retourne chaîne vide pour input vide/null', () => {
    expect(shortAddress('')).toBe('');
    expect(shortAddress(null)).toBe('');
    expect(shortAddress(undefined)).toBe('');
  });
});
