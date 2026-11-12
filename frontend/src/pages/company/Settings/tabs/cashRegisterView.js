/** Filtres, libellés et export. Aucune écriture et aucun total recalculé. */

const HISTORY_LABELS = {
  declare_cash: 'Encaissement déclaré',
  compensate_cash: 'Correction de saisie',
  declare_handover: 'Remise déclarée',
  confirm_handover: 'Remise confirmée',
  reconcile: 'Écart enregistré',
  regularize_variance: 'Régularisation',
  mark_uncertain: 'Réponse incertaine',
  refuse_payment: 'Paiement refusé',
};

const ACTION_ERRORS = {
  remise_ambigue: 'Plusieurs remises sont en attente. Confirmez-les une par une.',
  remise_non_confirmee: 'Cette remise n’est pas encore confirmée.',
  depassement_autorise: 'Le montant dépasse les espèces encore détenues.',
  conflit_idempotence: 'Cette opération existe déjà avec un autre montant.',
  rapprochement_deja_enregistre: 'Cet écart est déjà enregistré.',
  montant_invalide: 'Le montant saisi n’est pas valide.',
  montant_nul: 'Le montant doit être supérieur à zéro.',
};

export function cashHistoryLabel(action) {
  return HISTORY_LABELS[action] || 'Opération enregistrée';
}

export function handoverLabel(state) {
  if (state === 'declared') return 'Remise déclarée';
  if (state === 'confirmed') return 'Remise confirmée';
  return 'Aucune remise';
}

export function cashActionError(error) {
  const status = error?.response?.status;
  if (status === 403) return 'Cette action est réservée au responsable de l’entreprise.';
  if (status === 404) return 'Cette opération est indisponible pour votre entreprise.';
  const code = error?.response?.data?.error_code || error?.response?.data?.error;
  return ACTION_ERRORS[code] || 'L’opération n’a pas été enregistrée.';
}

export function allocationLabel(item) {
  if (!item) return 'Information d’imputation indisponible';
  if (item.allocation_status === 'unavailable') {
    return 'Information d’imputation indisponible';
  }
  if (item.invoice_number) return `Facture ${item.invoice_number}`;
  if (item.allocated_invoice_payment_id) return 'Imputé à une facture';
  if (item.allocation_status === 'allocated') return 'Imputé à une facture';
  return 'Non imputé';
}

export function filterCashCases(cases, filters) {
  const rows = Array.isArray(cases) ? cases : [];
  return rows.filter((item) => {
    if (filters.driver) {
      const needle = String(filters.driver).trim().toLowerCase();
      const label = String(item.driver_label || '').toLowerCase();
      if (String(item.driver_id) !== needle && !label.includes(needle)) return false;
    }
    if (filters.status && item.state !== filters.status) return false;
    if (filters.handover === 'aucune' && item.handover_state) return false;
    if (
      filters.handover
      && filters.handover !== 'aucune'
      && item.handover_state !== filters.handover
    ) {
      return false;
    }
    const variance = item.reconciliation_variance_chf;
    const hasGap = variance != null && variance !== '0.00';
    if (filters.discrepancy === 'oui' && !hasGap) return false;
    if (filters.discrepancy === 'non' && hasGap) return false;
    if (filters.operation && item.state !== filters.operation) return false;
    const day = String(item.created_at || '').slice(0, 10);
    if (filters.from && day && day < filters.from) return false;
    if (filters.to && day && day > filters.to) return false;
    return true;
  });
}

export function cashControlCsv(cases) {
  const lines = [
    'chauffeur;date;transport;recu;monnaie;net;detention;remise;rapprochement;exception;document;collection;remise_id;attendu;confirme;ecart;motif;statut',
  ];
  filterCashCases(cases, {}).forEach((item) => {
    const exception = item.state === 'administrative' || item.state === 'returned'
      ? 'administrative'
      : '';
    const details = Array.isArray(item.reconciliations) && item.reconciliations.length
      ? item.reconciliations
      : [null];
    details.forEach((detail) => {
      lines.push(
        [
          item.driver_id ?? '',
          String(item.created_at || '').slice(0, 10),
          item.booking_id ?? '',
          item.tendered_chf ?? '',
          item.change_chf ?? '',
          item.net_chf ?? '',
          item.state ?? '',
          item.handover_state ?? '',
          item.reconciliation_variance_chf ?? '',
          exception,
          'RECETTE — CONTROLE FICTIF',
          item.id ?? '',
          detail ? (detail.handover_id ?? '') : '',
          detail ? (detail.expected_chf ?? '') : '',
          detail ? (detail.counted_chf ?? '') : '',
          detail ? (detail.variance_chf ?? '') : '',
          detail ? String(detail.reason || '').replace(/;/g, ',') : '',
          detail ? (detail.status ?? '') : '',
        ].join(';')
      );
    });
  });
  return lines.join('\n');
}
