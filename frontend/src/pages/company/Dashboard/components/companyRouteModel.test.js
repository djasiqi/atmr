import {
  createInitialRoute,
  reorderRoutePoints,
  segmentLabels,
} from './companyRouteModel';

describe('companyRouteModel', () => {
  it('nomme les tronçons dans l ordre du parcours', () => {
    expect(segmentLabels(2, true)).toEqual([
      'Départ → Destination 1',
      'Destination 1 → Destination 2',
      'Destination 2 → Retour',
    ]);
    expect(segmentLabels(1, false)).toEqual(['Départ → Destination 1']);
  });

  it('réattribue le rôle selon la position après glisser-déposer', () => {
    const [pickup, first] = createInitialRoute('2026-09-28');
    const moved = reorderRoutePoints(
      [
        { ...pickup, location: 'Rue A' },
        { ...first, location: 'HUG', arrivalTime: '09:30' },
      ],
      0,
      1
    );
    expect(moved[0].role).toBe('pickup');
    expect(moved[0].location).toBe('HUG');
    expect(moved[0].arrivalTime).toBeUndefined();
    expect(moved[1].role).toBe('destination');
    expect(moved[1].location).toBe('Rue A');
  });
});
