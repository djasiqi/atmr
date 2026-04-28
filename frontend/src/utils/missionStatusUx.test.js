import { getMissionStatusPresentation } from './missionStatusUx';

describe('missionStatusUx', () => {
  it('retourne un libelle role-specifique', () => {
    const client = getMissionStatusPresentation('assigned', 'client');
    const company = getMissionStatusPresentation('assigned', 'company');
    expect(client.label).toBe('Chauffeur trouve');
    expect(company.label).toBe('Mission affectee');
  });

  it('retourne un statut inconnu si non mappe', () => {
    const result = getMissionStatusPresentation('mystery', 'client');
    expect(result.status).toBe('mystery');
    expect(result.label).toBe('Statut inconnu');
  });
});
