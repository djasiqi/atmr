/** Présentation UI du plan institution — aucune logique financière. */

export const MANUAL_FALLBACK_PAYER_TYPES = ['patient', 'clinic', 'partner'];

export const originLabel = (origin) => {
  const raw = String(origin || '').toLowerCase();
  if (raw === 'own_portfolio' || raw === 'portfolio') return 'Portefeuille';
  if (raw === 'lirie_marketplace' || raw === 'market_lirie' || raw === 'market') {
    return 'Market LIRIE';
  }
  return origin || '—';
};

export const validationLabel = (status) => {
  const raw = String(status || '').toLowerCase();
  if (raw === 'validated') return 'Validée';
  if (raw === 'auto_released') return 'Libérée à échéance';
  if (raw === 'pending' || raw === 'pending_review') return 'En attente';
  if (raw === 'disputed' || raw === 'anomaly') return 'Contestée';
  if (raw === 'validated_after_dispute') return 'Validée après contestation';
  if (raw === 'not_billable') return 'Non facturable';
  if (raw === 'not_required' || raw === 'own_portfolio') return 'Sans gate';
  return status || '—';
};

/** Libellé métier pour une prestation exclue — sans jargon Market / gate. */
export const exclusionDisplayLabel = (row) => {
  const bucket = String(row?.invoiceBucket || '').toLowerCase();
  const status = String(row?.validationStatus || '').toLowerCase();
  if (row?.exclusionReason === 'resolved_institution_not_billable' || status === 'not_billable') {
    return 'Exclue après contestation';
  }
  if (bucket === 'disputed_blocked' || status === 'disputed' || status === 'anomaly') {
    return "Contestée par l'institution";
  }
  if (bucket === 'pending_blocked' || status === 'pending' || status === 'pending_review') {
    return 'En attente de validation';
  }
  return 'Non facturable pour cette période';
};

/** Phrase qui explique pourquoi la course n'est pas dans le total. */
export const exclusionWhyText = (row) => {
  const bucket = String(row?.invoiceBucket || '').toLowerCase();
  const status = String(row?.validationStatus || '').toLowerCase();
  const disputeStatus = String(row?.disputeStatus || '').toLowerCase();
  if (disputeStatus === 'evidence_submitted') {
    return "Le transporteur a soumis un justificatif. La course reste hors facture jusqu'à validation de l'institution ou d'un opérateur LIRIE.";
  }
  if (disputeStatus === 'awaiting_carrier_response' || disputeStatus === 'awaiting_correction') {
    return "Le transporteur traite la contestation. La course reste hors du total tant que la résolution n'est pas validée.";
  }
  if (
    disputeStatus === 'disputed' ||
    row?.disputeTreatable ||
    bucket === 'disputed_blocked' ||
    status === 'disputed' ||
    status === 'anomaly'
  ) {
    return "L'institution a contesté cette course. Elle n'entre pas dans le total et ne sera pas facturée tant que la contestation n'est pas résolue.";
  }
  if (row?.exclusionReason === 'resolved_institution_not_billable' || status === 'not_billable') {
    return "Cette prestation a été exclue définitivement après contestation. La course reste dans l'historique et n'est pas facturée.";
  }
  if (bucket === 'pending_blocked' || status === 'pending' || status === 'pending_review') {
    return "Cette course n'est pas encore validée. Elle reste hors de cette facture jusqu'à validation ou libération à l'échéance.";
  }
  return "Cette course n'est pas facturable maintenant. Elle n'est pas comprise dans le montant ci-dessus.";
};

export const excludedBlockTitle = (count) =>
  Number(count) === 1
    ? "Pourquoi cette course n'est pas facturée"
    : 'Pourquoi ces courses ne sont pas facturées';

/** Identité lisible : patient, jamais le numéro interne de course. */
export const excludedRowWhoLabel = (row) =>
  String(row?.patientName || '').trim();

export const payerLabel = (payer) => {
  const raw = String(payer || '').toLowerCase();
  if (raw === 'clinic' || raw === 'institution') return 'Clinique';
  if (raw === 'patient') return 'Patient';
  if (raw === 'partner') return 'Partenaire';
  return payer || '—';
};

export const sumPatientBuckets = (patients = []) => ({
  patientCount: (patients || []).length,
  transportsCount: (patients || []).reduce(
    (n, p) => n + (Number(p.transports_count) || 0),
    0
  ),
  totalHt: (patients || []).reduce((n, p) => n + (Number(p.estimated_total) || 0), 0),
  bookingIds: (patients || []).flatMap((p) =>
    Array.isArray(p.booking_ids) ? p.booking_ids.map((id) => Number(id)) : []
  ),
});

const bucketOf = (plan, key) => plan?.reconciliation?.buckets?.[key] || {};

export const presentAutoDetectedPlan = (plan) => {
  const clinic = plan?.clinic || null;
  const patients = plan?.patients || [];
  const partners = plan?.partners || [];
  const pending = bucketOf(plan, 'pending_blocked');
  const disputed = bucketOf(plan, 'disputed_blocked');
  const clinicRecon = bucketOf(plan, 'clinic_billable');
  const partnerRecon = bucketOf(plan, 'partner_billable');
  const eligibility = plan?.eligibility || {};
  const market = eligibility.market_lirie || {};
  const patientStats = sumPatientBuckets(patients);
  const partnerStats = sumPatientBuckets(partners);
  const pendingCount = Number(pending.count ?? market.pending) || 0;
  const disputedCount = Number(disputed.count ?? market.disputed) || 0;

  return {
    clinic: {
      visible: Boolean(clinic && (Number(clinic.transports_count) || 0) > 0),
      displayName: clinic?.display_name || 'Clinique',
      transportsCount: Number(clinic?.transports_count) || 0,
      totalHt: Number(clinic?.estimated_total) || 0,
      bookingIds: Array.isArray(clinicRecon.booking_ids)
        ? clinicRecon.booking_ids.map((id) => Number(id))
        : Array.isArray(clinic?.booking_ids)
          ? clinic.booking_ids.map((id) => Number(id))
          : [],
    },
    patients: {
      visible: patientStats.patientCount > 0,
      ...patientStats,
    },
    partners: {
      visible: partnerStats.patientCount > 0 || (Number(partnerRecon.count) || 0) > 0,
      partnerCount: partnerStats.patientCount,
      transportsCount:
        partnerStats.transportsCount || (Number(partnerRecon.count) || 0),
      totalHt: partnerStats.totalHt || (Number(partnerRecon.amount_ht) || 0),
      bookingIds: Array.isArray(partnerRecon.booking_ids)
        ? partnerRecon.booking_ids.map((id) => Number(id))
        : partnerStats.bookingIds,
    },
    blocked: {
      visible: pendingCount + disputedCount > 0,
      billable: false,
      pendingCount,
      pendingAmountHt: Number(pending.amount_ht) || 0,
      pendingBookingIds: Array.isArray(pending.booking_ids)
        ? pending.booking_ids.map((id) => Number(id))
        : [],
      disputedCount,
      disputedAmountHt: Number(disputed.amount_ht) || 0,
      disputedBookingIds: Array.isArray(disputed.booking_ids)
        ? disputed.booking_ids.map((id) => Number(id))
        : [],
    },
    eligibility: {
      ownPortfolio: Number(eligibility.origin?.own_portfolio) || 0,
      marketLirie: Number(eligibility.origin?.market_lirie) || 0,
      validated: Number(market.validated) || 0,
      autoReleased: Number(market.auto_released) || 0,
    },
  };
};

export const presentAuditRows = (plan) =>
  (plan?.reconciliation?.bookings || []).map((row) => ({
    bookingId: Number(row.booking_id),
    origin: row.origin,
    originLabel: originLabel(row.origin),
    validationStatus: row.validation_status,
    validationLabel: validationLabel(row.validation_status),
    payer: row.payer,
    payerLabel: payerLabel(row.payer),
    eligible: Boolean(row.eligible),
    invoiceBucket: row.invoice_bucket,
    groupId: row.group_id || null,
    groupingRelation: row.grouping_relation || null,
    amountHt: Number(row.amount_ht) || 0,
    exclusionReason: row.exclusion_reason || null,
    disputeId: row.dispute_id != null ? Number(row.dispute_id) : null,
    disputeStatus: row.dispute_status ? String(row.dispute_status) : null,
    disputeTreatable: Boolean(row.dispute_treatable),
    invoiceLineId:
      row.invoice_line_id != null && Number.isFinite(Number(row.invoice_line_id))
        ? Number(row.invoice_line_id)
        : null,
    invoiceNumber: row.invoice_number ? String(row.invoice_number).trim() : '',
    patientName: String(row.patient_name || '').trim(),
    scheduledAt: row.scheduled_at || null,
    pickupLocation: String(row.pickup_location || '').trim(),
    dropoffLocation: String(row.dropoff_location || '').trim(),
  }));

export const alreadyInvoicedInvoiceLabel = (row) => {
  const number = String(row?.invoiceNumber || '').trim();
  if (number) return `Facture n° ${number}`;
  return 'Facture déjà émise';
};

export const groupAuditRowsForDisplay = (rows = []) => {
  const groups = new Map();
  const singles = [];
  (rows || []).forEach((row) => {
    if (row.groupId) {
      const key = `${row.groupId}:${row.payer || ''}`;
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(row);
      return;
    }
    singles.push({ key: `booking:${row.bookingId}`, expandable: false, rows: [row] });
  });
  const grouped = [];
  groups.forEach((members, key) => {
    grouped.push({
      key,
      expandable: members.length > 1,
      groupingRelation: members[0]?.groupingRelation || null,
      rows: members,
    });
  });
  return [...grouped, ...singles];
};

/** Résumé mode Institution : uniquement le payeur institution, pas les autres catégories. */
export const presentInstitutionInvoiceSummary = (plan) => {
  const clinic = plan?.clinic || null;
  const pending = bucketOf(plan, 'pending_blocked');
  const disputed = bucketOf(plan, 'disputed_blocked');
  const clinicRecon = bucketOf(plan, 'clinic_billable');
  const transportsCount =
    Number(clinic?.transports_count) || Number(clinicRecon.count) || 0;
  const totalHt = Number(clinic?.estimated_total) || Number(clinicRecon.amount_ht) || 0;
  const pendingCount = Number(pending.count) || 0;
  const disputedCount = Number(disputed.count) || 0;
  const alreadyRows = presentInstitutionAlreadyInvoicedRows(plan);
  const excludedRows = presentInstitutionExcludedRows(plan);
  const excludedCount = Math.max(pendingCount + disputedCount, excludedRows.length);
  const alreadyCount = alreadyRows.length;
  return {
    displayName: clinic?.display_name || 'Institution',
    transportsCount,
    totalHt,
    bookingIds: Array.isArray(clinicRecon.booking_ids)
      ? clinicRecon.booking_ids.map((id) => Number(id))
      : Array.isArray(clinic?.booking_ids)
        ? clinic.booking_ids.map((id) => Number(id))
        : [],
    hasBillable: transportsCount > 0,
    alreadyInvoiced: {
      visible: alreadyCount > 0,
      count: alreadyCount,
      amountHt: alreadyRows.reduce((n, row) => n + (row.amountHt || 0), 0),
      bookingIds: alreadyRows.map((row) => row.bookingId),
    },
    excluded: {
      visible: excludedCount > 0,
      count: excludedCount,
      amountHt:
        (Number(pending.amount_ht) || 0) + (Number(disputed.amount_ht) || 0),
      bookingIds: [
        ...(Array.isArray(pending.booking_ids) ? pending.booking_ids : []),
        ...(Array.isArray(disputed.booking_ids) ? disputed.booking_ids : []),
      ].map((id) => Number(id)),
    },
  };
};

export const presentInstitutionBillableRows = (plan) =>
  presentAuditRows(plan).filter(
    (row) =>
      row.invoiceBucket === 'clinic_billable' ||
      (row.payer === 'clinic' && row.eligible)
  );

const auditRouteDescription = (row) => {
  const pickup = String(row?.pickupLocation || '').trim();
  const dropoff = String(row?.dropoffLocation || '').trim();
  if (pickup && dropoff) return `Trajet : ${pickup} → ${dropoff}`;
  return '';
};

/** A/R d'affichage : parent_booking_id, jamais « même patient + même date ». */
export const groupInstitutionBillableRowsForPreview = (rows = []) => {
  const list = Array.isArray(rows) ? rows : [];
  const byId = new Map(list.map((row) => [Number(row.bookingId), row]));
  const used = new Set();
  const groups = [];

  const takePair = (first, second) => {
    used.add(Number(first.bookingId));
    used.add(Number(second.bookingId));
    groups.push({
      key: `ar:${first.bookingId}:${second.bookingId}`,
      expandable: true,
      groupingRelation: 'parent_booking_id',
      rows: [first, second],
    });
  };

  list.forEach((row) => {
    if (used.has(Number(row.bookingId))) return;
    const parentMatch = /^parent_booking_id:(\d+)$/u.exec(String(row.groupId || ''));
    if (!parentMatch) return;
    const parent = byId.get(Number(parentMatch[1]));
    if (!parent || used.has(Number(parent.bookingId))) return;
    takePair(parent, row);
  });

  return [...groups, ...groupAuditRowsForDisplay(list.filter((row) => !used.has(Number(row.bookingId))))];
};

/** Lignes d'aperçu Institution = seau clinique du plan, pas un second calcul. */
export const presentInstitutionBillablePreviewLines = (plan) => {
  const groups = groupInstitutionBillableRowsForPreview(
    presentInstitutionBillableRows(plan)
  );
  return groups
    .map((group) => {
      const members = [...(group.rows || [])].sort((a, b) => {
        const byDate = String(a.scheduledAt || '').localeCompare(
          String(b.scheduledAt || '')
        );
        if (byDate !== 0) return byDate;
        return (Number(a.bookingId) || 0) - (Number(b.bookingId) || 0);
      });
      const first = members[0];
      if (!first) return null;
      const last = members[members.length - 1];
      const isRoundTrip = Boolean(group.expandable && members.length > 1);
      return {
        preview_row_id: group.key,
        booking_id: first.bookingId,
        booking_ids: members.map((row) => row.bookingId),
        unit_type: isRoundTrip ? 'round_trip' : 'single',
        segments_count: members.length,
        is_round_trip_leg: isRoundTrip,
        round_trip_partner_booking_id: isRoundTrip ? last.bookingId : null,
        round_trip_primary_amount_ht: first.amountHt,
        round_trip_partner_amount_ht: isRoundTrip ? last.amountHt : null,
        scheduled_at: first.scheduledAt,
        patient_name: first.patientName,
        description: auditRouteDescription(first),
        round_trip_partner_description: isRoundTrip
          ? auditRouteDescription(last)
          : '',
        amount_ht: members.reduce((n, row) => n + (row.amountHt || 0), 0),
      };
    })
    .filter(Boolean)
    .sort((a, b) =>
      String(a.scheduled_at || '').localeCompare(String(b.scheduled_at || ''))
    );
};

export const presentInstitutionAlreadyInvoicedRows = (plan) =>
  presentAuditRows(plan).filter(
    (row) =>
      row.invoiceBucket === 'already_invoiced' &&
      (row.payer === 'clinic' || row.payer === 'institution' || !row.payer)
  );

export const presentInstitutionExcludedRows = (plan) =>
  presentAuditRows(plan).filter((row) => {
    if (row.invoiceBucket === 'already_invoiced') return false;
    return (
      row.invoiceBucket === 'pending_blocked' ||
      row.invoiceBucket === 'disputed_blocked' ||
      (row.payer === 'clinic' && !row.eligible)
    );
  });

/** Parité : le plan auto affiche exactement les buckets certifiés, sans recalcul. */
export const autoPlanParityWithCertifiedBuckets = (plan) => {
  const presented = presentAutoDetectedPlan(plan);
  const clinicCount = Number(plan?.clinic?.transports_count) || 0;
  const clinicHt = Number(plan?.clinic?.estimated_total) || 0;
  const patientStats = sumPatientBuckets(plan?.patients || []);
  return {
    clinicCountMatch: presented.clinic.transportsCount === clinicCount,
    clinicHtMatch: presented.clinic.totalHt === clinicHt,
    patientCountMatch: presented.patients.patientCount === patientStats.patientCount,
    patientTransportsMatch:
      presented.patients.transportsCount === patientStats.transportsCount,
    patientHtMatch: presented.patients.totalHt === patientStats.totalHt,
    patientBookingIdsMatch:
      JSON.stringify(presented.patients.bookingIds) ===
      JSON.stringify(patientStats.bookingIds),
    blockedNotBillable: presented.blocked.billable === false,
  };
};

export const flattenInstitutionPreviewBookingIds = (lines = []) =>
  (Array.isArray(lines) ? lines : []).flatMap((line) =>
    Array.isArray(line?.booking_ids)
      ? line.booking_ids.map((id) => Number(id)).filter((id) => Number.isFinite(id))
      : [Number(line?.booking_id)].filter((id) => Number.isFinite(id))
  );

/** Résumé + lignes = deux projections du même plan, même rendu React. */
export const institutionSurfacesFromPlan = (plan) => {
  const summary = presentInstitutionInvoiceSummary(plan);
  const lines = presentInstitutionBillablePreviewLines(plan);
  const excluded = presentInstitutionExcludedRows(plan);
  const lineBookingIds = flattenInstitutionPreviewBookingIds(lines);
  return { summary, lines, excluded, lineBookingIds };
};

const sortedIds = (ids = []) =>
  [...ids].map((id) => Number(id)).filter((id) => Number.isFinite(id)).sort((a, b) => a - b);

/** Booking ids portés par un brouillon (meta, racine, ou lignes). */
export const collectDraftInvoiceBookingIds = (draft) => {
  const fromRoot = draft?.booking_ids || draft?.reservation_ids;
  if (Array.isArray(fromRoot) && fromRoot.length) {
    return fromRoot.map((id) => Number(id)).filter((id) => Number.isFinite(id));
  }
  const meta =
    draft?.meta && typeof draft.meta === 'object' && !Array.isArray(draft.meta)
      ? draft.meta
      : {};
  const fromMeta = meta.booking_ids || meta.reservation_ids;
  if (Array.isArray(fromMeta) && fromMeta.length) {
    return fromMeta.map((id) => Number(id)).filter((id) => Number.isFinite(id));
  }
  const lines = Array.isArray(draft?.lines) ? draft.lines : [];
  const ids = [];
  for (const line of lines) {
    const extra = line?.line_meta?.booking_ids || line?.booking_ids;
    if (Array.isArray(extra) && extra.length) {
      extra.forEach((id) => {
        const n = Number(id);
        if (Number.isFinite(n)) ids.push(n);
      });
      continue;
    }
    const rid = line?.reservation_id ?? line?.booking_id;
    const n = Number(rid);
    if (Number.isFinite(n)) ids.push(n);
  }
  return ids;
};

export const draftInvoiceAmountHt = (draft) => {
  const raw = draft?.total_ht ?? draft?.subtotal_ht ?? draft?.amount_ht ?? draft?.total;
  const n = Number(raw);
  return Number.isFinite(n) ? n : null;
};

/** UI-DRAFT-6 / 7 : le brouillon préparé doit coller au dernier plan. */
export const draftInvoicePlanParity = (draft, plan) => {
  const { summary } = institutionSurfacesFromPlan(plan);
  const planIds = sortedIds(summary.bookingIds);
  const draftIds = sortedIds(collectDraftInvoiceBookingIds(draft));
  const draftTotal = draftInvoiceAmountHt(draft);
  return {
    planTotal: summary.totalHt,
    draftTotal,
    planBookingIds: planIds,
    draftBookingIds: draftIds,
    totalMatch: draftTotal != null && Math.abs(draftTotal - summary.totalHt) < 0.005,
    bookingIdsMatch: JSON.stringify(planIds) === JSON.stringify(draftIds),
  };
};

/** UI-1 / UI-14 : les deux surfaces lisent les mêmes booking_ids et le même HT. */
export const institutionSurfaceParity = (plan) => {
  const { summary, lines, excluded, lineBookingIds } = institutionSurfacesFromPlan(plan);
  const sourceIds = sortedIds(summary.bookingIds);
  const fromLines = sortedIds(lineBookingIds);
  const excludedIds = new Set(sortedIds(excluded.map((row) => row.bookingId)));
  const lineHt = lines.reduce((n, line) => n + (Number(line.amount_ht) || 0), 0);
  return {
    bookingIdsMatch: JSON.stringify(sourceIds) === JSON.stringify(fromLines),
    excludedOutsideBillable: fromLines.every((id) => !excludedIds.has(id)),
    lineCountLeqPrestations: lines.length <= summary.transportsCount,
    prestationCountMatch: fromLines.length === summary.transportsCount,
    totalHtMatch: lineHt === summary.totalHt,
    lineHt,
    summaryHt: summary.totalHt,
  };
};
