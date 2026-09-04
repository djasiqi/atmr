import {
  defaultSelectedPatientIds,
  selectedPatientStats,
  summarizePatientBatchResult,
} from '../institutionPatientBatchUi';

describe('institutionPatientBatchUi', () => {
  const patients = [
    { institution_patient_id: 1, transports_count: 1, estimated_total: 40 },
    { institution_patient_id: 2, transports_count: 2, estimated_total: 80 },
    { institution_patient_id: 3, transports_count: 1, estimated_total: 40 },
  ];

  it('sélectionne tous les débiteurs par défaut', () => {
    expect(defaultSelectedPatientIds(patients)).toEqual([1, 2, 3]);
  });

  it('un patient décoché n’entre pas dans les stats envoyées', () => {
    expect(selectedPatientStats(patients, [1, 3])).toEqual({
      patientCount: 2,
      transportsCount: 2,
      totalHt: 80,
    });
  });

  it('résume created / reused / failed après génération', () => {
    const summary = summarizePatientBatchResult({
      data: {
        requested_patient_count: 14,
        created_count: 12,
        reused_count: 2,
        failed_count: 0,
        invoices: [{ invoice_id: 1 }, { invoice_id: 2 }],
      },
    });
    expect(summary).toMatchObject({
      created: 12,
      reused: 2,
      failed: 0,
      requested: 14,
      hasErrors: false,
    });
    expect(summary.invoices).toHaveLength(2);
  });
});
