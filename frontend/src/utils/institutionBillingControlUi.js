/**
 * Présentation pure — contrôle facturation institution (sans logique métier).
 */

const MONTH_LABELS = [
  'Janvier',
  'Février',
  'Mars',
  'Avril',
  'Mai',
  'Juin',
  'Juillet',
  'Août',
  'Septembre',
  'Octobre',
  'Novembre',
  'Décembre',
];

export function defaultPeriodValue(date = new Date()) {
  const y = date.getFullYear();
  const m = String(date.getMonth() + 1).padStart(2, '0');
  return `${y}-${m}`;
}

export function parsePeriodValue(periodValue) {
  if (!periodValue || typeof periodValue !== 'string') return null;
  const m = periodValue.match(/^(\d{4})-(\d{2})$/);
  if (!m) return null;
  const year = Number(m[1]);
  const month = Number(m[2]);
  if (month < 1 || month > 12) return null;
  return { year, month };
}

export function formatPeriodLabel(periodValue) {
  const parsed = parsePeriodValue(periodValue);
  if (!parsed) return periodValue || '';
  return `${MONTH_LABELS[parsed.month - 1]} ${parsed.year}`;
}

export function buildBillingControlQueryParams(filters = {}) {
  const params = {
    page: filters.page || 1,
    page_size: filters.page_size || 50,
  };
  if (filters.period) params.period = filters.period;
  if (filters.control_status) params.control_status = filters.control_status;
  if (filters.payer_type) params.payer_type = filters.payer_type;
  if (filters.transport_company) params.transport_company = filters.transport_company;
  if (filters.patient) params.patient = filters.patient;
  return params;
}

export function segmentTypeLabel(segmentType) {
  const map = {
    outbound: 'Aller',
    return: 'Retour',
    segment: 'Segment',
  };
  return map[segmentType] || segmentType || 'Trajet';
}

export function payerTypeLabel(payerType) {
  const t = String(payerType || '').toLowerCase();
  if (t === 'clinic') return 'Clinique';
  if (t === 'patient') return 'Patient';
  return payerType || '—';
}

export function billingIntentFromPayerType(payerType) {
  return String(payerType || '').toLowerCase() === 'clinic' ? 'institution' : 'patient';
}

export function controlStatusLabel(status) {
  const map = {
    pending_review: 'À vérifier',
    validated: 'Validé',
    anomaly: 'Anomalie',
    auto_released: 'Libérée à échéance',
    disputed: 'Contestée',
  };
  return map[status] || status || '—';
}

export function formatBookingDate(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '—';
  return d.toLocaleDateString('fr-CH', { day: '2-digit', month: '2-digit' });
}

export function bookingDateKey(iso) {
  if (!iso) return 'unknown';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return 'unknown';
  return d.toISOString().slice(0, 10);
}

export function groupBookingsForDisplay(items = []) {
  const groups = new Map();
  for (const item of items) {
    const patientName = item?.patient?.display_name || 'Patient';
    const dateKey = bookingDateKey(item?.scheduled_time || item?.date);
    const groupKey = `${patientName}::${dateKey}`;
    if (!groups.has(groupKey)) {
      groups.set(groupKey, {
        key: groupKey,
        patientName,
        dateKey,
        dateLabel: formatBookingDate(item?.scheduled_time || item?.date),
        items: [],
      });
    }
    groups.get(groupKey).items.push(item);
  }
  return Array.from(groups.values()).sort((a, b) => {
    if (a.dateKey === b.dateKey) {
      return a.patientName.localeCompare(b.patientName, 'fr');
    }
    return a.dateKey.localeCompare(b.dateKey);
  });
}

export function collectTransportCompanyOptions(items = []) {
  const map = new Map();
  for (const item of items) {
    const tc = item?.transport_company;
    if (tc?.company_id && tc?.display_name) {
      map.set(String(tc.company_id), tc.display_name);
    }
  }
  return Array.from(map.entries())
    .map(([value, label]) => ({ value, label }))
    .sort((a, b) => a.label.localeCompare(b.label, 'fr'));
}

export function parseBillingControlApiError(error) {
  const status = error?.response?.status;
  const message = error?.response?.data?.error
    || error?.message
    || 'Une erreur est survenue.';
  if (status === 403) {
    return 'Accès refusé. Votre rôle ne permet pas le contrôle facturation.';
  }
  if (status === 409) {
    return message;
  }
  return message;
}

export function isBookingEditable(item) {
  return Boolean(item?.billing?.editable);
}

export function isBookingLocked(item) {
  return Boolean(item?.billing?.locked || item?.billing?.invoiced);
}

const OPEN_DISPUTE_STATUSES = new Set([
  'disputed',
  'awaiting_carrier_response',
  'evidence_submitted',
  'awaiting_correction',
]);

export function isOpenDisputeStatus(status) {
  return OPEN_DISPUTE_STATUSES.has(String(status || ''));
}

export function canDecideDispute(item) {
  return String(item?.control?.dispute_status || '') === 'evidence_submitted';
}

export function isFinanciallyFrozen(item) {
  if (isOpenDisputeStatus(item?.control?.dispute_status)) return true;
  const billing = String(item?.control?.invoice_billing_status || '');
  return (
    item?.control?.effective_status === 'anomaly' && billing !== 'not_billable'
  );
}

const SUMMARY_STATUS_KEYS = new Set(['pending_review', 'validated', 'anomaly']);

export function normalizeControlStatus(status) {
  const value = String(status || '').toLowerCase();
  if (value === 'validated') return 'validated';
  if (value === 'anomaly') return 'anomaly';
  return 'pending_review';
}

export function sameBookingId(left, right) {
  return String(left) === String(right);
}

export function adjustBillingControlSummary(summary = {}, fromStatus, toStatus) {
  const from = normalizeControlStatus(fromStatus);
  const to = normalizeControlStatus(toStatus);
  const next = {
    total: summary.total ?? 0,
    payer_clinic: summary.payer_clinic ?? 0,
    payer_patient: summary.payer_patient ?? 0,
    validated: summary.validated ?? 0,
    pending_review: summary.pending_review ?? 0,
    anomaly: summary.anomaly ?? 0,
  };
  if (from === to) return next;
  if (SUMMARY_STATUS_KEYS.has(from)) {
    next[from] = Math.max(0, (next[from] ?? 0) - 1);
  }
  if (SUMMARY_STATUS_KEYS.has(to)) {
    next[to] = (next[to] ?? 0) + 1;
  }
  return next;
}

export function applyOptimisticControlStatus(
  listData,
  bookingId,
  nextStatus,
  controlPatch = {},
) {
  if (!listData || !Array.isArray(listData.items)) return listData;
  let fromStatus = null;
  const items = listData.items.map((item) => {
    if (!sameBookingId(item.booking_id, bookingId)) return item;
    fromStatus = item?.control?.effective_status || 'pending_review';
    return {
      ...item,
      control: {
        ...item.control,
        effective_status: nextStatus,
        ...controlPatch,
      },
    };
  });
  if (fromStatus == null) return listData;
  return {
    ...listData,
    items,
    summary: adjustBillingControlSummary(listData.summary, fromStatus, nextStatus),
  };
}

export function applyOptimisticControlOverrides(listData, overrides = {}) {
  if (!listData || !overrides || Object.keys(overrides).length === 0) {
    return listData;
  }
  return Object.entries(overrides).reduce((acc, [bookingId, override]) => {
    if (!override) return acc;
    let next = acc;
    if (override.payer) {
      next = applyOptimisticPayerChange(next, bookingId, override.payer);
    }
    if (override.status) {
      next = applyOptimisticControlStatus(
        next,
        bookingId,
        override.status,
        override.controlPatch,
      );
    }
    return next;
  }, listData);
}

export function mergeBookingControlFromMutation(listData, bookingId, control) {
  if (!control) return listData;
  const nextStatus = normalizeControlStatus(
    control.effective_status || control.control_status,
  );
  return applyOptimisticControlStatus(listData, bookingId, nextStatus, {
    validated_at: control.validated_at,
    validated_by_display_name: control.validated_by_display_name,
    anomaly_reason: control.anomaly_reason ?? null,
  });
}

export function pruneSyncedControlOverrides(overrides = {}, items = []) {
  const next = { ...overrides };
  let changed = false;
  Object.keys(next).forEach((bookingId) => {
    const item = items.find((candidate) => sameBookingId(candidate.booking_id, bookingId));
    if (!item) {
      delete next[bookingId];
      changed = true;
      return;
    }
    const override = next[bookingId] || {};
    const statusSynced = !override.status
      || item?.control?.effective_status === override.status;
    const payerSynced = !override.payer
      || normalizePayerType(item?.payer?.type) === override.payer;
    if (statusSynced && payerSynced) {
      delete next[bookingId];
      changed = true;
      return;
    }
    if (statusSynced || payerSynced) {
      const cleaned = { ...override };
      if (statusSynced) {
        delete cleaned.status;
        delete cleaned.fromStatus;
        delete cleaned.optimistic;
        delete cleaned.controlPatch;
      }
      if (payerSynced) {
        delete cleaned.payer;
        delete cleaned.fromPayer;
        delete cleaned.payerGeneration;
      }
      next[bookingId] = cleaned;
      changed = true;
    }
  });
  return changed ? next : overrides;
}

export function normalizePayerType(payerType) {
  return String(payerType || '').toLowerCase() === 'clinic' ? 'clinic' : 'patient';
}

export function adjustBillingControlPayerSummary(summary = {}, fromPayer, toPayer) {
  const from = normalizePayerType(fromPayer);
  const to = normalizePayerType(toPayer);
  const next = {
    total: summary.total ?? 0,
    payer_clinic: summary.payer_clinic ?? 0,
    payer_patient: summary.payer_patient ?? 0,
    validated: summary.validated ?? 0,
    pending_review: summary.pending_review ?? 0,
    anomaly: summary.anomaly ?? 0,
  };
  if (from === to) return next;
  if (from === 'clinic') next.payer_clinic = Math.max(0, (next.payer_clinic ?? 0) - 1);
  if (from === 'patient') next.payer_patient = Math.max(0, (next.payer_patient ?? 0) - 1);
  if (to === 'clinic') next.payer_clinic = (next.payer_clinic ?? 0) + 1;
  if (to === 'patient') next.payer_patient = (next.payer_patient ?? 0) + 1;
  return next;
}

export function applyOptimisticPayerChange(listData, bookingId, nextPayerType) {
  if (!listData || !Array.isArray(listData.items)) return listData;
  const nextPayer = normalizePayerType(nextPayerType);
  let fromPayer = null;
  const items = listData.items.map((item) => {
    if (!sameBookingId(item.booking_id, bookingId)) return item;
    fromPayer = normalizePayerType(item?.payer?.type);
    return {
      ...item,
      payer: {
        ...item.payer,
        type: nextPayer,
        display_name: payerTypeLabel(nextPayer),
      },
    };
  });
  if (fromPayer == null) return listData;
  return {
    ...listData,
    items,
    summary: adjustBillingControlPayerSummary(listData.summary, fromPayer, nextPayer),
  };
}

export function applyOptimisticPayerMutation(listData, bookingId, nextPayerType) {
  let next = applyOptimisticPayerChange(listData, bookingId, nextPayerType);
  const item = next?.items?.find((candidate) => sameBookingId(candidate.booking_id, bookingId));
  if (item?.control?.effective_status !== 'validated') return next;
  return applyOptimisticControlStatus(next, bookingId, 'pending_review', {
    validated_at: null,
    validated_by_display_name: null,
  });
}

export function isCanceledApiError(error) {
  return Boolean(
    error?.code === 'ERR_CANCELED'
    || error?.name === 'CanceledError'
    || error?.name === 'AbortError',
  );
}
