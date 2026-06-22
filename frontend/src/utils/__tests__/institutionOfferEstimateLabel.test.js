import { institutionOfferEstimateLabel } from '../institutionOfferEstimateLabel';

describe('institutionOfferEstimateLabel', () => {
  it('affiche Tarif préférentiel uniquement si institution + source preferential', () => {
    expect(
      institutionOfferEstimateLabel({ source: 'preferential' }, 'institution'),
    ).toBe('Tarif préférentiel');
  });

  it('n’affiche pas préférentiel si billing patient malgré source preferential', () => {
    expect(
      institutionOfferEstimateLabel({ source: 'preferential' }, 'patient'),
    ).toBe('Tarif estimé');
  });

  it('affiche le libellé profil entreprise pour company_profile', () => {
    expect(
      institutionOfferEstimateLabel({ source: 'company_profile' }, 'patient'),
    ).toBe('Tarif estimé (profil tarifaire)');
  });

  it('accepte le legacy source profile', () => {
    expect(
      institutionOfferEstimateLabel({ source: 'profile' }, 'patient'),
    ).toBe('Tarif estimé (profil tarifaire)');
  });
});
