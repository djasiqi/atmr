import { patientCandidatesForPeriod } from '../invoiceService';

describe('patientCandidatesForPeriod', () => {
  const september = {
    data: {
      period: '2026-09',
      patients: [{ id: 'client:1|billing_party:2', name: 'Jean Dupont' }],
    },
  };

  it('ignore une réponse tardive d’un autre mois', () => {
    expect(patientCandidatesForPeriod(september, 2026, 7)).toEqual([]);
    expect(patientCandidatesForPeriod(september, 2026, 9)).toEqual(
      september.data.patients,
    );
  });
});
