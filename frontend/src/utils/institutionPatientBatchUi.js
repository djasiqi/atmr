/** Helpers UI — sélection et résumé du batch factures patients. */

export const defaultSelectedPatientIds = (patients = []) =>
  (patients || [])
    .map((p) => Number(p?.institution_patient_id))
    .filter((id) => Number.isFinite(id) && id > 0);

export const selectedPatientBuckets = (patients = [], selectedIds) => {
  const selected = new Set((selectedIds || []).map((id) => Number(id)));
  return (patients || []).filter((p) => selected.has(Number(p?.institution_patient_id)));
};

export const selectedPatientStats = (patients = [], selectedIds) => {
  const buckets = selectedPatientBuckets(patients, selectedIds);
  return {
    patientCount: buckets.length,
    transportsCount: buckets.reduce((n, p) => n + (Number(p.transports_count) || 0), 0),
    totalHt: buckets.reduce((n, p) => n + (Number(p.estimated_total) || 0), 0),
  };
};

export const unwrapInstitutionPatientBatch = (res) => {
  if (res && typeof res === 'object' && res.data && typeof res.data === 'object') {
    if ('created_count' in res.data || 'invoices' in res.data) {
      return res.data;
    }
  }
  return res;
};

export const summarizePatientBatchResult = (raw) => {
  const result = unwrapInstitutionPatientBatch(raw) || {};
  const created = Number(result.created_count) || 0;
  const reused = Number(result.reused_count) || 0;
  const failed = Number(result.failed_count) || 0;
  const skipped = Number(result.skipped_count) || 0;
  const invoices = Array.isArray(result.invoices) ? result.invoices : [];
  return {
    created,
    reused,
    failed,
    skipped,
    requested: Number(result.requested_patient_count ?? result.patient_count) || 0,
    invoices,
    hasErrors: failed > 0,
  };
};
