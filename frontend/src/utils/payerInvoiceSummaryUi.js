/** Présentation UI Patient / Partenaire — aucune logique financière. */

export const presentPatientInvoiceSummary = (opportunity) => {
  if (!opportunity) {
    return {
      visible: false,
      hasBillable: false,
      displayName: '',
      transportsCount: 0,
      totalHt: 0,
      blocked: false,
    };
  }
  const transportsCount =
    Number(opportunity.transports_count) ||
    Number(opportunity.segments_count) ||
    Number(opportunity.units_count) ||
    0;
  const totalHt = Number(opportunity.unbilled_total_amount) || 0;
  const blocked = opportunity.can_generate === false;
  const displayName =
    opportunity.display_name ||
    `${opportunity.first_name || ''} ${opportunity.last_name || ''}`.trim() ||
    'Patient';
  return {
    visible: true,
    displayName,
    transportsCount,
    totalHt,
    hasBillable: !blocked && (transportsCount > 0 || totalHt > 0),
    blocked,
  };
};

export const presentPartnerInvoiceSummary = (row) => {
  if (!row) {
    return {
      visible: false,
      hasBillable: false,
      displayName: '',
      transportsCount: 0,
      totalHt: 0,
      excluded: { visible: false, count: 0 },
    };
  }
  const transportsCount = Number(row.validated_unbilled_transfers_count) || 0;
  const totalHt = Number(row.estimated_subtotal_ht ?? row.total_amount) || 0;
  const unbilled = Number(row.unbilled_transfers_count) || 0;
  const excludedCount = Math.max(0, unbilled - transportsCount);
  return {
    visible: true,
    displayName: row.partner_company_name || 'Partenaire',
    transportsCount,
    totalHt,
    hasBillable: transportsCount > 0,
    excluded: {
      visible: excludedCount > 0,
      count: excludedCount,
    },
  };
};
