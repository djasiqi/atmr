/**
 * Contrôle facturation institution — présentation pure sur le contrat API (INSTITUTION-07).
 */

import React, { useCallback, useMemo, useState } from 'react';
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
import {
  buildBillingControlQueryParams,
  collectTransportCompanyOptions,
  controlStatusLabel,
  defaultPeriodValue,
  formatPeriodLabel,
  groupBookingsForDisplay,
  isBookingEditable,
  isBookingLocked,
  parseBillingControlApiError,
  payerTypeLabel,
  billingIntentFromPayerType,
  segmentTypeLabel,
  formatBookingDate,
} from '../../../utils/institutionBillingControlUi';
import s from './InstitutionBillingControl.module.css';

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
    return (
      <div>
        <span className={s.statusAnomaly}>⚠ Anomalie</span>
        {item?.control?.anomaly_reason && (
          <div className={s.anomalyReason}>Motif : {item.control.anomaly_reason}</div>
        )}
      </div>
    );
  }
  return <span className={s.statusPending}>{controlStatusLabel(status)}</span>;
}

function BookingActions({
  item,
  onValidate,
  onAnomaly,
  onReopen,
  pendingId,
}) {
  if (isBookingLocked(item) || !isBookingEditable(item)) return null;
  const status = item?.control?.effective_status;
  const busy = pendingId === item.booking_id;

  if (status === 'anomaly' || status === 'validated') {
    return (
      <div className={s.actions}>
        <button
          type="button"
          className={s.btn}
          disabled={busy}
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
        disabled={busy}
        onClick={() => onValidate(item)}
      >
        ✓ Valider
      </button>
      <button
        type="button"
        className={`${s.btn} ${s.btnDanger}`}
        disabled={busy}
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
  const [pendingId, setPendingId] = useState(null);

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
    refetch,
    isFetching,
  } = useBillingControlBookings(queryParams, allowed);

  const { data: patientsData } = useInstitutionPatients({ per_page: 200 }, allowed);
  const patients = patientsData?.patients || patientsData?.items || [];

  const validateMutation = useValidateBillingControlBooking();
  const anomalyMutation = useMarkBillingControlAnomaly();
  const reopenMutation = useReopenBillingControlBooking();
  const payerMutation = useChangeBillingControlPayer();

  const items = useMemo(() => data?.items ?? [], [data?.items]);
  const summary = data?.summary || {};
  const pagination = data?.pagination || {};
  const groups = useMemo(() => groupBookingsForDisplay(items), [items]);
  const transportOptions = useMemo(
    () => collectTransportCompanyOptions(items),
    [items],
  );

  const handleMutationError = useCallback((err) => {
    toast.error(parseBillingControlApiError(err));
  }, []);

  const runMutation = useCallback(async (bookingId, fn) => {
    setPendingId(bookingId);
    try {
      await fn();
      await refetch();
    } catch (err) {
      handleMutationError(err);
    } finally {
      setPendingId(null);
    }
  }, [refetch, handleMutationError]);

  const handlePayerChange = useCallback((item, newPayerType) => {
    if (!isBookingEditable(item)) return;
    const current = String(item?.payer?.type || '').toLowerCase();
    if (current === newPayerType) return;
    runMutation(item.booking_id, () => payerMutation.mutateAsync({
      bookingId: item.booking_id,
      data: {
        billing_intent: billingIntentFromPayerType(newPayerType),
        billing_change_reason_code: 'ADMIN_CORRECTION',
        override_reason: 'Correction payeur — contrôle facturation',
      },
    }).then(() => {
      toast.success('Payeur mis à jour — statut repassé à « À vérifier »');
    }));
  }, [payerMutation, runMutation]);

  const handleValidate = useCallback((item) => {
    runMutation(item.booking_id, () => validateMutation.mutateAsync({
      bookingId: item.booking_id,
      data: {},
    }).then(() => toast.success('Booking validé')));
  }, [validateMutation, runMutation]);

  const handleReopen = useCallback((item) => {
    runMutation(item.booking_id, () => reopenMutation.mutateAsync({
      bookingId: item.booking_id,
      data: {},
    }).then(() => toast.success('Anomalie levée — à vérifier')));
  }, [reopenMutation, runMutation]);

  const submitAnomaly = useCallback(() => {
    if (!anomalyTarget) return;
    const reason = anomalyReason.trim();
    if (!reason) {
      toast.error('Indiquez un motif.');
      return;
    }
    runMutation(anomalyTarget.booking_id, () => anomalyMutation.mutateAsync({
      bookingId: anomalyTarget.booking_id,
      data: {
        anomaly_reason_code: 'OTHER',
        comment: reason,
      },
    }).then(() => {
      toast.success('Anomalie signalée');
      setAnomalyTarget(null);
      setAnomalyReason('');
    }));
  }, [anomalyTarget, anomalyReason, anomalyMutation, runMutation]);

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
                  <th>Date</th>
                  <th>Patient</th>
                  <th>Trajet</th>
                  <th>Transporteur</th>
                  <th>Payeur</th>
                  <th>Contrôle</th>
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
                      <tr key={item.booking_id} data-booking-id={item.booking_id}>
                        <td>{formatBookingDate(item.scheduled_time)}</td>
                        <td>{item.patient?.display_name || '—'}</td>
                        <td>
                          <span className={s.segmentLabel}>
                            {segmentTypeLabel(item.segment_type)}
                          </span>
                          <span className={s.routeHint}>
                            {item.pickup || '—'} → {item.dropoff || '—'}
                          </span>
                        </td>
                        <td>{item.transport_company?.display_name || '—'}</td>
                        <td>
                          {isBookingEditable(item) ? (
                            <select
                              className={s.payerSelect}
                              data-testid={`payer-select-${item.booking_id}`}
                              value={String(item.payer?.type || 'patient').toLowerCase() === 'clinic' ? 'clinic' : 'patient'}
                              disabled={pendingId === item.booking_id || isFetching}
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
                        <td>
                          <ControlStatusCell item={item} />
                          <BookingActions
                            item={item}
                            pendingId={pendingId}
                            onValidate={handleValidate}
                            onAnomaly={setAnomalyTarget}
                            onReopen={handleReopen}
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
                disabled={pendingId === anomalyTarget.booking_id}
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
