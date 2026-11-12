function openBalance(value) {
  if (value == null || value === '' || value === '0' || value === '0.00') return false;
  return true;
}

export function cashBusinessState(row) {
  if (!row) return 'Vérification nécessaire';
  if (row.state === 'prepared') return 'À encaisser';
  if (row.state === 'received') {
    return openBalance(row.balance_chf) ? 'Partiellement réglé' : 'Réglé';
  }
  return 'Vérification nécessaire';
}

export function dossierBusinessState(row) {
  if (!row) return 'À facturer';
  if (row.invoice_issued === true || row.invoice_issuance === true) return 'Facture émise';
  if (row.draft_invoice_id || row.state === 'preparing' || row.state === 'prepared') {
    return 'Facture préparée';
  }
  return 'À facturer';
}

export function mandateBusinessState(row) {
  if (!row || row.status === 'demo_absent') return 'Aucun mandat';
  const signed = row.signature_recorded
    || row.specimen_recorded
    || row.status === 'demo_specimen_signed'
    || row.status === 'demo_valid_simulated';
  if (!signed) return 'Mandat en brouillon';
  const archived = row.demonstration_signature?.documentary_status === 'archived'
    || row.documentary_status === 'archived';
  return archived
    ? 'Version signée et archivée localement'
    : 'Version signée, document non archivé';
}
