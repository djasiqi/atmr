/** Présentation UI Patient / Partenaire — aucune logique financière. */

const GATE_HELD_REASONS = new Set(['pending_institution_validation', 'disputed']);

const plural = (count, singular, pluralForm) => (count === 1 ? singular : pluralForm);

/** ISO (instant Europe/Zurich) → « JJ.MM.AAAA », ou null si invalide. */
export const formatReleaseDateCH = (iso) => {
  if (!iso) return null;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return null;
  try {
    return d.toLocaleDateString('fr-CH', {
      timeZone: 'Europe/Zurich',
      day: '2-digit',
      month: '2-digit',
      year: 'numeric',
    });
  } catch {
    return null;
  }
};

/** Phrase « N prestation(s) en attente de validation par l'institution … ». */
export const pendingValidationSentence = ({ count, releaseAtLabel }) => {
  if (!count) return '';
  const head = `${count} ${plural(count, 'prestation', 'prestations')} en attente de validation par l'institution`;
  const tail = releaseAtLabel
    ? ` — ${plural(count, 'facturable', 'facturables')} après validation ou automatiquement dès le ${releaseAtLabel}.`
    : ` — ${plural(count, 'facturable', 'facturables')} après validation ou à la fin du mois.`;
  return `${head}${tail}`;
};

const disputedSentence = (count) =>
  count
    ? `${count} ${plural(count, 'prestation contestée', 'prestations contestées')} par l'institution — à traiter avant facturation.`
    : '';

export const presentPatientInvoiceSummary = (opportunity) => {
  if (!opportunity) {
    return {
      visible: false,
      hasBillable: false,
      displayName: '',
      transportsCount: 0,
      totalHt: 0,
      blocked: false,
      blockedReason: null,
      gateHeld: false,
      pendingValidation: { visible: false, count: 0, amountHt: 0, releaseAtLabel: null },
      disputed: { visible: false, count: 0 },
      emptyNote: '',
      pendingNote: '',
      invoiceDeliveryMethod: 'email',
    };
  }
  const transportsCount =
    Number(opportunity.transports_count) ||
    Number(opportunity.segments_count) ||
    Number(opportunity.units_count) ||
    0;
  const totalHt = Number(opportunity.unbilled_total_amount) || 0;
  const pendingCount = Number(opportunity.pending_validation_count) || 0;
  const pendingAmount = Number(opportunity.pending_validation_amount) || 0;
  const releaseAtLabel = formatReleaseDateCH(opportunity.pending_validation_release_at);
  const disputedCount = Number(opportunity.disputed_count) || 0;

  const cannotGenerate = opportunity.can_generate === false;
  const backendReason = opportunity.blocked_reason || null;
  const blockedReason = backendReason || (cannotGenerate ? 'incomplete' : null);
  const gateHeld = GATE_HELD_REASONS.has(blockedReason);
  // « blocked » garde son sens historique : identité / destinataire à compléter.
  const blocked = cannotGenerate && !gateHeld;

  const displayName =
    opportunity.display_name ||
    `${opportunity.first_name || ''} ${opportunity.last_name || ''}`.trim() ||
    'Patient';
  const rawDelivery = String(opportunity.invoice_delivery_method || 'email')
    .trim()
    .toLowerCase();

  const hasBillable = !cannotGenerate && (transportsCount > 0 || totalHt > 0);

  let emptyNote = '';
  if (!hasBillable) {
    if (blockedReason === 'pending_institution_validation') {
      emptyNote = pendingValidationSentence({ count: pendingCount, releaseAtLabel });
    } else if (blockedReason === 'disputed') {
      emptyNote = disputedSentence(disputedCount);
    } else if (blocked) {
      emptyNote = 'Identité ou destinataire à compléter avant facturation.';
    } else {
      emptyNote = 'Aucune prestation à facturer à ce patient pour cette période.';
    }
  }

  // Cas mixte : une partie facturable maintenant, une partie retenue.
  const pendingNote =
    hasBillable && pendingCount > 0
      ? `${pendingValidationSentence({ count: pendingCount, releaseAtLabel }).replace(/\.$/, '')} — non ${plural(pendingCount, 'incluse', 'incluses')} dans cette facture.`
      : '';

  return {
    visible: true,
    displayName,
    transportsCount,
    totalHt,
    hasBillable,
    blocked,
    blockedReason,
    gateHeld,
    pendingValidation: {
      visible: pendingCount > 0,
      count: pendingCount,
      amountHt: pendingAmount,
      releaseAtLabel,
    },
    disputed: { visible: disputedCount > 0, count: disputedCount },
    emptyNote,
    pendingNote,
    invoiceDeliveryMethod: rawDelivery === 'paper' ? 'paper' : 'email',
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
