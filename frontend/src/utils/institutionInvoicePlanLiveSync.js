/** Synchronisation live du plan institution — pas de logique financière. */

export const INSTITUTION_PLAN_POLL_MS = 10000;

export const institutionPlanScopeKey = ({
  clinicCompanyId,
  periodYear,
  periodMonth,
}) => `${clinicCompanyId || ''}|${periodYear || ''}|${periodMonth || ''}`;

export const nextInstitutionPlanRequestId = (current) =>
  (Number(current) || 0) + 1;

export const shouldApplyInstitutionPlanResponse = (reqId, latestReqId) =>
  Number(reqId) === Number(latestReqId);

export const isInstitutionPlanFetchCanceled = (err) => {
  const name = String(err?.name || '');
  const code = String(err?.code || '');
  return (
    name === 'CanceledError' ||
    name === 'AbortError' ||
    code === 'ERR_CANCELED' ||
    err?.silent === true
  );
};

/** Garde le dernier plan valide : pas de flash 0 / 0 sur un refresh raté. */
export const shouldKeepPreviousInstitutionPlan = ({
  silent,
  hasPreviousPlan,
} = {}) =>
  Boolean(silent) || Boolean(hasPreviousPlan);

export const shouldShowInstitutionPlanSkeleton = ({
  loading,
  planIsCurrent,
} = {}) =>
  Boolean(loading) && !planIsCurrent;

export const isInstitutionPlanCurrent = (plan, planScope, currentScope) =>
  Boolean(plan) && String(planScope || '') === String(currentScope || '');

export const shouldPollInstitutionPlan = ({
  open,
  payerType,
  clinicCompanyId,
  visibilityState,
} = {}) =>
  Boolean(open) &&
  payerType === 'clinic' &&
  Boolean(clinicCompanyId) &&
  visibilityState === 'visible';

export const clinicMonthlyPreparePayload = ({
  clinicCompanyId,
  periodYear,
  periodMonth,
}) => ({
  mode: 'clinic_monthly',
  clinic_company_id: clinicCompanyId,
  period_year: periodYear,
  period_month: periodMonth,
});

export const unwrapPreparedDraftInvoice = (result) => {
  let inv = result?.data ?? result;
  if (inv?.data?.id != null) inv = inv.data;
  if (!inv?.id) return null;
  return { ...inv, status: inv.status || 'draft' };
};

export const draftInvoiceFromPrepareError = (err) => {
  const data = err?.response?.data || err?.data || {};
  const id = data.existing_invoice_id ?? data.invoice_id;
  if (id == null) return null;
  return {
    id,
    invoice_number: data.existing_invoice_number || data.invoice_number || '',
    status: 'draft',
  };
};

/** Toolbar brouillon : uniquement après Prepare, jamais dans le résumé ni la preview lignes. */
export const shouldShowDraftInvoiceToolbar = ({ hasPreparedDraft } = {}) =>
  Boolean(hasPreparedDraft);

export const shouldShowSimpleInvoiceLinesPreview = ({
  showLinesPreview,
  hasPreparedDraft,
} = {}) => Boolean(showLinesPreview) && !hasPreparedDraft;

/** Trois états UX du modal : résumé / lignes simples / brouillon. */
export const presentBillPeriodComposerUi = ({
  hasPreparedDraft,
  showLinesPreview,
} = {}) => ({
  showDraftToolbar: shouldShowDraftInvoiceToolbar({ hasPreparedDraft }),
  showSimpleLinesPreview: shouldShowSimpleInvoiceLinesPreview({
    showLinesPreview,
    hasPreparedDraft,
  }),
  showPrepareFooter: !hasPreparedDraft,
});

/** Parité DraftInvoiceEditorPanel : actions PDF selon le lifecycle du brouillon. */
export const draftPdfActionsAvailability = ({
  invoiceStatus,
  hasStoredPdf,
} = {}) => {
  const status = String(invoiceStatus || '').toLowerCase();
  const allowsLineEditing = ['draft', 'sent', 'partially_paid', 'overdue'].includes(
    status
  );
  return {
    download: allowsLineEditing || Boolean(hasStoredPdf),
    print: true,
    open: Boolean(hasStoredPdf),
  };
};

/**
 * Écoute socket + visibility + poll. Un seul abonnement ; cleanup au démontage.
 * Le poll est actif seulement si l’onglet est visible.
 */
export const bindInstitutionPlanLiveRefresh = ({
  socket,
  refresh,
  pollMs = INSTITUTION_PLAN_POLL_MS,
  documentRef = typeof document !== 'undefined' ? document : null,
  setIntervalFn = typeof window !== 'undefined'
    ? window.setInterval.bind(window)
    : setInterval,
  clearIntervalFn = typeof window !== 'undefined'
    ? window.clearInterval.bind(window)
    : clearInterval,
} = {}) => {
  if (typeof refresh !== 'function') return () => {};

  let timer = null;
  const onSocket = () => refresh({ reason: 'socket' });
  const syncPoll = () => {
    if (timer != null) {
      clearIntervalFn(timer);
      timer = null;
    }
    if (documentRef?.visibilityState === 'visible') {
      timer = setIntervalFn(() => refresh({ reason: 'poll' }), pollMs);
    }
  };
  const onVisibility = () => {
    if (documentRef?.visibilityState === 'visible') {
      refresh({ reason: 'visibility' });
    }
    syncPoll();
  };

  if (socket?.on) {
    socket.on('booking_updated', onSocket);
  }
  if (documentRef?.addEventListener) {
    documentRef.addEventListener('visibilitychange', onVisibility);
  }
  syncPoll();

  return () => {
    if (timer != null) {
      clearIntervalFn(timer);
      timer = null;
    }
    if (documentRef?.removeEventListener) {
      documentRef.removeEventListener('visibilitychange', onVisibility);
    }
    if (socket?.off) {
      socket.off('booking_updated', onSocket);
    }
  };
};
