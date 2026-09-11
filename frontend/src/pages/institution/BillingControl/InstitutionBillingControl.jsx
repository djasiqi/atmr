/**
 * Contrôle facturation institution — présentation pure sur le contrat API (INSTITUTION-07).
 */

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Navigate } from 'react-router-dom';
import { toast } from 'sonner';
import Modal from '../../../components/common/Modal';
import {
  useBillingControlBookings,
  useChangeBillingControlPayer,
  useInstitutionMe,
  useInstitutionPatients,
  useMarkBillingControlAnomaly,
  useReopenBillingControlBooking,
  useValidateBillingControlBooking,
} from '../../../hooks/useInstitutionData';
import { canAccessBillingControl } from '../../../utils/institutionPermissions';
import institutionBillingControlService from '../../../services/institutionBillingControlService';
import {
  buildBillingControlQueryParams,
  canDecideDispute,
  collectTransportCompanyOptions,
  controlStatusLabel,
  defaultPeriodValue,
  formatPeriodLabel,
  groupBookingsForDisplay,
  isBookingEditable,
  isBookingLocked,
  isFinanciallyFrozen,
  parseBillingControlApiError,
  payerTypeLabel,
  billingIntentFromPayerType,
  segmentTypeLabel,
  formatBookingDate,
  applyOptimisticControlOverrides,
  pruneSyncedControlOverrides,
  normalizePayerType,
  isCanceledApiError,
} from '../../../utils/institutionBillingControlUi';
import {
  markBillingPayerChange,
  markBillingPayerUiCommit,
  markBillingValidateClick,
  markBillingValidateUiCommit,
} from '../../../utils/billingControlValidatePerf';
import s from './InstitutionBillingControl.module.css';

const VALIDATE_ERROR_MESSAGE = "La validation n'a pas pu être enregistrée. Réessayez.";
const PAYER_ERROR_MESSAGE = "Le payeur n'a pas pu être modifié. Réessayez.";
const REOPEN_ERROR_MESSAGE = "La réouverture n'a pas pu être enregistrée. Réessayez.";
const ANOMALY_ERROR_MESSAGE = "L'anomalie n'a pas pu être enregistrée. Réessayez.";

const STATUS_FILTER_OPTIONS = [
  { value: '', label: 'Tous' },
  { value: 'pending_review', label: 'À vérifier' },
  { value: 'validated', label: 'Validé' },
  { value: 'anomaly', label: 'Anomalie' },
];

const PAYER_FILTER_OPTIONS = [
  { value: '', label: 'Tous' },
  { value: 'patient', label: 'Patient' },
  { value: 'clinic', label: 'Clinique' },
];

function ControlStatusCell({ item }) {
  const status = item?.control?.effective_status;
  if (isBookingLocked(item)) {
    return (
      <div>
        <span className={s.lockedBadge}>🔒 Facturé</span>
        <div className={s.payerReadonly}>
          Payeur : {payerTypeLabel(item?.payer?.type)}
        </div>
      </div>
    );
  }
  if (status === 'validated') {
    return (
      <div>
        <span className={s.statusValidated}>✓ Validé</span>
        {item?.control?.validated_by_display_name && (
          <div className={s.validatedMeta}>
            par {item.control.validated_by_display_name}
          </div>
        )}
        {item?.control?.validated_at && (
          <div className={s.validatedMeta}>
            {new Date(item.control.validated_at).toLocaleString('fr-CH')}
          </div>
        )}
      </div>
    );
  }
  if (status === 'anomaly') {
    const disputeStatus = item?.control?.dispute_status;
    return (
      <div>
        <span className={s.statusAnomaly}>⚠ Anomalie</span>
        {item?.control?.anomaly_reason && (
          <div className={s.anomalyReason}>Motif : {item.control.anomaly_reason}</div>
        )}
        {disputeStatus === 'evidence_submitted' ? (
          <div className={s.anomalyReason}>Justificatif soumis — à valider</div>
        ) : null}
      </div>
    );
  }
  if (item?.control?.invoice_gate_status === 'auto_released') {
    return (
      <span className={s.statusPending} title="En attente à la clôture — libérée pour facturation">
        Libérée à échéance
      </span>
    );
  }
  return <span className={s.statusPending}>{controlStatusLabel(status)}</span>;
}

function BookingActions({
  item,
  onValidate,
  onAnomaly,
  onReopen,
  onAcceptEvidence,
  onRejectEvidence,
  hideReopen,
}) {
  if (isBookingLocked(item) || !isBookingEditable(item)) return null;
  const status = item?.control?.effective_status;

  if (canDecideDispute(item)) {
    return (
      <div className={s.actions}>
        <button
          type="button"
          className={`${s.btn} ${s.btnPrimary}`}
          data-testid={`dispute-accept-${item.booking_id}`}
          onClick={() => onAcceptEvidence(item)}
        >
          Valider le justificatif
        </button>
        <button
          type="button"
          className={s.btn}
          data-testid={`dispute-reject-${item.booking_id}`}
          onClick={() => onRejectEvidence(item)}
        >
          Refuser
        </button>
      </div>
    );
  }

  if (status === 'anomaly' || status === 'validated') {
    if (hideReopen || isFinanciallyFrozen(item)) return null;
    return (
      <div className={s.actions}>
        <button
          type="button"
          className={s.btn}
          onClick={() => onReopen(item)}
        >
          Réouvrir
        </button>
      </div>
    );
  }

  return (
    <div className={s.actions}>
      <button
        type="button"
        className={`${s.btn} ${s.btnPrimary}`}
        onClick={() => onValidate(item)}
      >
        ✓ Valider
      </button>
      <button
        type="button"
        className={`${s.btn} ${s.btnDanger}`}
        onClick={() => onAnomaly(item)}
      >
        ⚠ Signaler une anomalie
      </button>
    </div>
  );
}

const InstitutionBillingControl = () => {
  const { data: meData } = useInstitutionMe();
  const institutionRole = meData?.institution_role;
  const allowed = canAccessBillingControl(institutionRole);

  const [period, setPeriod] = useState(defaultPeriodValue());
  const [controlStatus, setControlStatus] = useState('');
  const [payerType, setPayerType] = useState('');
  const [transportCompany, setTransportCompany] = useState('');
  const [patientId, setPatientId] = useState('');
  const [page, setPage] = useState(1);
  const [anomalyTarget, setAnomalyTarget] = useState(null);
  const [anomalyReason, setAnomalyReason] = useState('');
  const [rowOverrides, setRowOverrides] = useState({});
  const inFlightValidateRef = useRef(new Set());
  const inFlightReopenRef = useRef(new Set());
  const payerGenerationRef = useRef({});
  const payerAbortRef = useRef({});

  const queryParams = useMemo(
    () => buildBillingControlQueryParams({
      period,
      control_status: controlStatus,
      payer_type: payerType,
      transport_company: transportCompany || undefined,
      patient: patientId || undefined,
      page,
      page_size: 50,
    }),
    [period, controlStatus, payerType, transportCompany, patientId, page],
  );

  const {
    data,
    isLoading,
    isError,
    error,
  } = useBillingControlBookings(queryParams, allowed);

  const { data: patientsData } = useInstitutionPatients({ per_page: 200 }, allowed);
  const patients = patientsData?.patients || patientsData?.items || [];

  const validateMutation = useValidateBillingControlBooking();
  const anomalyMutation = useMarkBillingControlAnomaly();
  const reopenMutation = useReopenBillingControlBooking();
  const payerMutation = useChangeBillingControlPayer();

  const serverItems = useMemo(() => data?.items ?? [], [data?.items]);

  useEffect(() => {
    setRowOverrides((prev) => pruneSyncedControlOverrides(prev, serverItems));
  }, [serverItems]);

  const displayData = useMemo(
    () => applyOptimisticControlOverrides(
      { items: serverItems, summary: data?.summary || {} },
      rowOverrides,
    ),
    [serverItems, data?.summary, rowOverrides],
  );
  const items = displayData?.items ?? serverItems;
  const summary = displayData?.summary || {};
  const pagination = data?.pagination || {};
  const groups = useMemo(() => groupBookingsForDisplay(items), [items]);
  const transportOptions = useMemo(
    () => collectTransportCompanyOptions(items),
    [items],
  );

  const handleMutationError = useCallback((err) => {
    toast.error(parseBillingControlApiError(err));
  }, []);

  const runBackgroundMutation = useCallback((fn) => {
    Promise.resolve()
      .then(fn)
      .catch(handleMutationError);
  }, [handleMutationError]);

  const handlePayerChange = useCallback((item, newPayerType) => {
    if (!isBookingEditable(item)) return;
    const bookingId = item.booking_id;
    const nextPayer = normalizePayerType(newPayerType);
    const current = normalizePayerType(item?.payer?.type);
    if (current === nextPayer) return;

    const generation = (payerGenerationRef.current[bookingId] || 0) + 1;
    payerGenerationRef.current[bookingId] = generation;
    payerAbortRef.current[bookingId]?.abort();
    const abortController = typeof AbortController !== 'undefined'
      ? new AbortController()
      : null;
    if (abortController) {
      payerAbortRef.current[bookingId] = abortController;
    }

    const clickAt = markBillingPayerChange(bookingId);
    const wasValidated = item?.control?.effective_status === 'validated';
    setRowOverrides((prev) => ({
      ...prev,
      [bookingId]: {
        ...prev[bookingId],
        payer: nextPayer,
        fromPayer: current,
        payerGeneration: generation,
        ...(wasValidated
          ? { status: 'pending_review', fromStatus: 'validated', optimistic: false }
          : {}),
      },
    }));
    markBillingPayerUiCommit(bookingId, clickAt);

    payerMutation.mutateAsync({
      bookingId,
      payerType: nextPayer,
      generation,
      clickAt,
      signal: abortController?.signal,
      data: {
        billing_intent: billingIntentFromPayerType(nextPayer),
        billing_change_reason_code: 'ADMIN_CORRECTION',
        override_reason: 'Correction payeur — contrôle facturation',
      },
    }).catch((err) => {
      if (isCanceledApiError(err)) return;
      if (payerGenerationRef.current[bookingId] !== generation) return;
      setRowOverrides((prev) => {
        const currentOverride = prev[bookingId];
        if (!currentOverride) return prev;
        const next = { ...prev };
        const restored = { ...currentOverride, payer: current };
        delete restored.fromPayer;
        delete restored.payerGeneration;
        if (wasValidated) {
          restored.status = 'validated';
          delete restored.fromStatus;
        }
        if (!restored.status && !restored.payer) {
          delete next[bookingId];
        } else {
          next[bookingId] = restored;
        }
        return next;
      });
      toast.error(PAYER_ERROR_MESSAGE);
    });
  }, [payerMutation]);

  const handleValidate = useCallback((item) => {
    const bookingId = item.booking_id;
    const fromStatus = item?.control?.effective_status || 'pending_review';
    if (fromStatus === 'validated') return;
    if (inFlightValidateRef.current.has(bookingId)) return;

    const clickAt = markBillingValidateClick(bookingId);
    inFlightValidateRef.current.add(bookingId);
    setRowOverrides((prev) => {
      if (prev[bookingId]?.status === 'validated') return prev;
      return {
        ...prev,
        [bookingId]: { status: 'validated', fromStatus, optimistic: true },
      };
    });
    markBillingValidateUiCommit(bookingId, clickAt);

    validateMutation.mutateAsync({
      bookingId,
      data: {},
      clickAt,
    }).catch(() => {
      setRowOverrides((prev) => {
        if (!prev[bookingId]) return prev;
        const next = { ...prev };
        delete next[bookingId];
        return next;
      });
      toast.error(VALIDATE_ERROR_MESSAGE);
    }).finally(() => {
      inFlightValidateRef.current.delete(bookingId);
    });
  }, [validateMutation]);

  const handleReopen = useCallback((item) => {
    const bookingId = item.booking_id;
    const fromStatus = item?.control?.effective_status;
    if (fromStatus !== 'validated' && fromStatus !== 'anomaly') return;
    if (inFlightReopenRef.current.has(bookingId)) return;

    inFlightReopenRef.current.add(bookingId);
    setRowOverrides((prev) => ({
      ...prev,
      [bookingId]: {
        ...prev[bookingId],
        status: 'pending_review',
        fromStatus,
        optimistic: false,
        controlPatch: {
          validated_at: null,
          validated_by_display_name: null,
          anomaly_reason: null,
        },
      },
    }));

    reopenMutation.mutateAsync({
      bookingId,
      data: {},
    }).catch(() => {
      setRowOverrides((prev) => {
        const currentOverride = prev[bookingId];
        if (!currentOverride) return prev;
        const next = { ...prev };
        next[bookingId] = {
          ...currentOverride,
          status: fromStatus,
          fromStatus: undefined,
          controlPatch: undefined,
        };
        return next;
      });
      toast.error(REOPEN_ERROR_MESSAGE);
    }).finally(() => {
      inFlightReopenRef.current.delete(bookingId);
    });
  }, [reopenMutation]);

  const handleAcceptEvidence = useCallback((item) => {
    runBackgroundMutation(() =>
      institutionBillingControlService.decideBillingControlDispute(item.booking_id, {
        decision: 'accept_carrier',
      }).then(() => toast.success('Justificatif validé — prestation à nouveau facturable')),
    );
  }, [runBackgroundMutation]);

  const handleRejectEvidence = useCallback((item) => {
    runBackgroundMutation(() =>
      institutionBillingControlService.decideBillingControlDispute(item.booking_id, {
        decision: 'reject_evidence',
      }).then(() => toast.success('Justificatif refusé — le transporteur doit compléter')),
    );
  }, [runBackgroundMutation]);

  const submitAnomaly = useCallback(() => {
    if (!anomalyTarget) return;
    const reason = anomalyReason.trim();
    if (!reason) {
      toast.error('Indiquez un motif.');
      return;
    }
    const bookingId = anomalyTarget.booking_id;
    const fromStatus = anomalyTarget?.control?.effective_status || 'pending_review';
    setAnomalyTarget(null);
    setAnomalyReason('');
    setRowOverrides((prev) => ({
      ...prev,
      [bookingId]: {
        ...prev[bookingId],
        status: 'anomaly',
        fromStatus,
        optimistic: false,
        controlPatch: { anomaly_reason: `OTHER: ${reason}` },
      },
    }));
    anomalyMutation.mutateAsync({
      bookingId,
      data: {
        anomaly_reason_code: 'OTHER',
        comment: reason,
      },
    }).catch(() => {
      setRowOverrides((prev) => {
        const currentOverride = prev[bookingId];
        if (!currentOverride) return prev;
        const next = { ...prev };
        next[bookingId] = {
          ...currentOverride,
          status: fromStatus,
          fromStatus: undefined,
          controlPatch: undefined,
        };
        return next;
      });
      toast.error(ANOMALY_ERROR_MESSAGE);
    });
  }, [anomalyTarget, anomalyReason, anomalyMutation]);

  if (!allowed) {
    if (meData && !canAccessBillingControl(institutionRole)) {
      return (
        <div className={s.forbidden} data-testid="billing-control-forbidden">
          Accès refusé. Cette page est réservée aux rôles Administrateur et Facturation.
        </div>
      );
    }
    return null;
  }

  if (error?.response?.status === 403) {
    return (
      <div className={s.forbidden} data-testid="billing-control-api-403">
        Accès refusé par le serveur.
      </div>
    );
  }

  return (
    <div className={s.page} data-testid="billing-control-page">
      <section className={s.summaryCard}>
        <h2 className={s.periodTitle}>{formatPeriodLabel(period)}</h2>
        <div className={s.summaryGrid} data-testid="billing-control-summary">
          <span className={s.summaryItem}>
            <strong>{summary.total ?? 0}</strong> trajets à contrôler
          </span>
          <span className={s.summaryItem}>
            <strong>{summary.payer_clinic ?? 0}</strong> Clinique
          </span>
          <span className={s.summaryItem}>
            <strong>{summary.payer_patient ?? 0}</strong> Patient
          </span>
          <span className={s.summaryItem}>
            <strong>{summary.validated ?? 0}</strong> Validés
          </span>
          <span className={s.summaryItem}>
            <strong>{summary.pending_review ?? 0}</strong> À vérifier
          </span>
          <span className={s.summaryItem}>
            <strong>{summary.anomaly ?? 0}</strong> Anomalies
          </span>
        </div>
      </section>

      <section className={s.filters} data-testid="billing-control-filters">
        <div className={s.filterField}>
          <label htmlFor="bc-period">Période</label>
          <input
            id="bc-period"
            type="month"
            value={period}
            onChange={(e) => { setPeriod(e.target.value); setPage(1); }}
          />
        </div>
        <div className={s.filterField}>
          <label htmlFor="bc-status">Statut</label>
          <select
            id="bc-status"
            value={controlStatus}
            onChange={(e) => { setControlStatus(e.target.value); setPage(1); }}
          >
            {STATUS_FILTER_OPTIONS.map((o) => (
              <option key={o.value || 'all'} value={o.value}>{o.label}</option>
            ))}
          </select>
        </div>
        <div className={s.filterField}>
          <label htmlFor="bc-payer">Filtre payeur</label>
          <select
            id="bc-payer"
            value={payerType}
            onChange={(e) => { setPayerType(e.target.value); setPage(1); }}
          >
            {PAYER_FILTER_OPTIONS.map((o) => (
              <option key={o.value || 'all'} value={o.value}>{o.label}</option>
            ))}
          </select>
        </div>
        <div className={s.filterField}>
          <label htmlFor="bc-carrier">Transporteur</label>
          <select
            id="bc-carrier"
            value={transportCompany}
            onChange={(e) => { setTransportCompany(e.target.value); setPage(1); }}
          >
            <option value="">Tous</option>
            {transportOptions.map((o) => (
              <option key={o.value} value={o.value}>{o.label}</option>
            ))}
          </select>
        </div>
        <div className={s.filterField}>
          <label htmlFor="bc-patient">Patient</label>
          <select
            id="bc-patient"
            value={patientId}
            onChange={(e) => { setPatientId(e.target.value); setPage(1); }}
          >
            <option value="">Tous</option>
            {patients.map((p) => (
              <option key={p.id} value={p.id}>
                {[p.first_name, p.last_name].filter(Boolean).join(' ') || `#${p.id}`}
              </option>
            ))}
          </select>
        </div>
      </section>

      {isLoading && (
        <div className={s.loading} data-testid="billing-control-loading">
          Chargement…
        </div>
      )}

      {isError && !isLoading && (
        <div className={s.error} data-testid="billing-control-error">
          {parseBillingControlApiError(error)}
        </div>
      )}

      {!isLoading && !isError && items.length === 0 && (
        <div className={s.empty} data-testid="billing-control-empty">
          Aucun transport à contrôler pour cette période.
        </div>
      )}

      {!isLoading && !isError && items.length > 0 && (
        <>
          <div className={s.tableWrap}>
            <table className={s.table} data-testid="billing-control-table">
              <thead>
                <tr>
                  <th className={s.colDate}>Date</th>
                  <th className={s.colPatient}>Patient</th>
                  <th className={s.colRoute}>Trajet</th>
                  <th className={s.colCarrier}>Transporteur</th>
                  <th className={s.colPayer}>Payeur</th>
                  <th className={s.colControl}>Contrôle</th>
                </tr>
              </thead>
              <tbody>
                {groups.map((group) => (
                  <React.Fragment key={group.key}>
                    <tr className={s.groupRow} data-testid="billing-control-group">
                      <td colSpan={6}>
                        {group.patientName} — {group.dateLabel}
                      </td>
                    </tr>
                    {group.items.map((item) => (
                      <tr
                        key={item.booking_id}
                        className={s.bookingRow}
                        data-booking-id={item.booking_id}
                      >
                        <td className={s.colDate} data-label="Date">
                          {formatBookingDate(item.scheduled_time)}
                        </td>
                        <td className={`${s.colPatient} ${s.patientCell}`} data-label="Patient">
                          {item.patient?.display_name || '—'}
                        </td>
                        <td className={s.colRoute} data-label="Trajet">
                          <span className={s.segmentLabel}>
                            {segmentTypeLabel(item.segment_type)}
                          </span>
                          <span className={s.routeHint}>
                            {item.pickup || '—'} → {item.dropoff || '—'}
                          </span>
                        </td>
                        <td className={s.colCarrier} data-label="Transporteur">
                          {item.transport_company?.display_name || '—'}
                        </td>
                        <td className={s.colPayer} data-label="Payeur">
                          {isBookingEditable(item) && !isFinanciallyFrozen(item) ? (
                            <select
                              className={s.payerSelect}
                              data-testid={`payer-select-${item.booking_id}`}
                              value={normalizePayerType(item.payer?.type)}
                              onChange={(e) => handlePayerChange(item, e.target.value)}
                              aria-label={`Payeur booking ${item.booking_id}`}
                            >
                              <option value="patient">Patient</option>
                              <option value="clinic">Clinique</option>
                            </select>
                          ) : (
                            <span className={s.payerReadonly}>
                              {payerTypeLabel(item.payer?.type)}
                            </span>
                          )}
                        </td>
                        <td className={s.colControl} data-label="Contrôle">
                          <ControlStatusCell item={item} />
                          <BookingActions
                            item={item}
                            hideReopen={Boolean(rowOverrides[item.booking_id]?.optimistic)}
                            onValidate={handleValidate}
                            onAnomaly={setAnomalyTarget}
                            onReopen={handleReopen}
                            onAcceptEvidence={handleAcceptEvidence}
                            onRejectEvidence={handleRejectEvidence}
                          />
                        </td>
                      </tr>
                    ))}
                  </React.Fragment>
                ))}
              </tbody>
            </table>
          </div>

          {(pagination.total_pages || 1) > 1 && (
            <div className={s.pagination} data-testid="billing-control-pagination">
              <button
                type="button"
                className={s.btn}
                disabled={page <= 1}
                onClick={() => setPage((p) => Math.max(1, p - 1))}
              >
                Précédent
              </button>
              <span>
                Page {pagination.page || page} / {pagination.total_pages || 1}
                {' '}
                ({pagination.total ?? summary.total ?? 0} trajets)
              </span>
              <button
                type="button"
                className={s.btn}
                disabled={page >= (pagination.total_pages || 1)}
                onClick={() => setPage((p) => p + 1)}
              >
                Suivant
              </button>
            </div>
          )}
        </>
      )}

      {anomalyTarget && (
        <Modal onClose={() => setAnomalyTarget(null)} size="md" ariaLabel="Signaler une anomalie">
          <h3>Signaler une anomalie</h3>
          <div className={s.modalBody}>
            <label htmlFor="anomaly-reason">Motif</label>
            <textarea
              id="anomaly-reason"
              value={anomalyReason}
              onChange={(e) => setAnomalyReason(e.target.value)}
              placeholder="Décrivez le problème…"
            />
            <div className={s.modalActions}>
              <button type="button" className={s.btn} onClick={() => setAnomalyTarget(null)}>
                Annuler
              </button>
              <button
                type="button"
                className={`${s.btn} ${s.btnDanger}`}
                onClick={submitAnomaly}
              >
                Signaler
              </button>
            </div>
          </div>
        </Modal>
      )}
    </div>
  );
};

/** Garde route — redirige si rôle non autorisé (navigation directe). */
export function BillingControlRouteGuard({ children }) {
  const { data: meData, isLoading } = useInstitutionMe();
  if (isLoading) return null;
  if (!canAccessBillingControl(meData?.institution_role)) {
    return <Navigate to="/unauthorized" replace />;
  }
  return children;
}

export default InstitutionBillingControl;
