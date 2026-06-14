// pages/institution/Requests/InstitutionRequestDetail.jsx
/**
 * Fiche détaillée d'un transport — design professionnel SaaS B2B
 */

import React, { useState, useCallback, useMemo, useEffect } from 'react';
import ConfirmSendModal from './ConfirmSendModal';
import { useParams, useNavigate } from 'react-router-dom';
import {
  FaArrowLeft, FaEdit, FaPaperPlane, FaTimes,
  FaTruck, FaRoute, FaFileInvoiceDollar,
  FaHistory, FaWheelchair, FaInfoCircle, FaNotesMedical,
  FaEnvelope,
} from 'react-icons/fa';
import {
  useInstitutionRequest, useInstitutionMe,
  useSendRequest, useCancelRequest,
  useAssignExternalCarrier, useCompleteExternalMission,
  useUpdateRequestBilling, useUpdateBookingBilling,
} from '../../../hooks/useInstitutionData';
import ExternalCarrierFields, {
  EMPTY_EXTERNAL_CARRIER_FORM,
  validateExternalCarrierForm,
  buildExternalCarrierPayload,
} from '../../../components/institution/ExternalCarrierFields';
import {
  isExternalRequest,
  hasBooking,
  canAssignExternalCarrier,
  canCompleteExternalMission,
  getRequestStatusLabel,
  EXTERNAL_STATUSES,
} from '../../../utils/requestStatus';
import { canManageRequests, canEditBilling } from '../../../utils/institutionPermissions';
import { toast } from 'sonner';
import { getInstitutionSocket } from '../../../services/institutionSocket';
import { fetchBookingMessages, sendBookingMessage, exportRequestMissionPdf } from '../../../services/institutionService';
import { buildCarrierMailto } from '../../../utils/externalCarrierEmail';
import BookingChat from '../../company/Reservations/components/BookingChat';
import s from './InstitutionRequestDetail.module.css';

// ─── Status mapping ────────────────────────────────────────
const BOOKING_STATUS_MAP = {
  PENDING:          { label: 'En attente',           css: 'statusPending' },
  ACCEPTED:         { label: 'Accepté',              css: 'statusAccepted' },
  ASSIGNED:         { label: 'Chauffeur assigné',    css: 'statusAssigned' },
  EN_ROUTE:         { label: 'En route',             css: 'statusEnRoute' },
  IN_PROGRESS:      { label: 'En cours',             css: 'statusInProgress' },
  OUTBOUND_COMPLETED: { label: 'Retour en cours',    css: 'statusInProgress' },
  COMPLETED:        { label: 'Terminé',              css: 'statusCompleted' },
  RETURN_COMPLETED: { label: 'Aller-retour terminé', css: 'statusReturnCompleted' },
  CANCELED:         { label: 'Annulé',               css: 'statusCancelled' },
};

const REQUEST_STATUS_MAP = {
  DRAFT:     { label: 'Brouillon', css: 'statusDraft' },
  SENT:      { label: 'Envoyée',   css: 'statusSent' },
  ACCEPTED:  { label: 'Acceptée',  css: 'statusAccepted' },
  CONVERTED: { label: 'Confirmée', css: 'statusConverted' },
  CANCELLED: { label: 'Annulée',   css: 'statusCancelled' },
  EXPIRED:   { label: 'Expirée',   css: 'statusExpired' },
  [EXTERNAL_STATUSES.ASSIGNED]: {
    label: 'Transporteur externe affecté',
    css: 'statusExternalAssigned',
  },
  [EXTERNAL_STATUSES.COMPLETED]: {
    label: 'Déclarée réalisée par l\'institution',
    css: 'statusExternalCompleted',
  },
};

const MISSION_LABELS = {
  patient_transport: 'Transport patient',
  material_delivery: 'Livraison matériel',
};

// ─── Helpers ───────────────────────────────────────────────
const fmt = (dateStr) => {
  if (!dateStr) return '—';
  return new Date(dateStr).toLocaleString('fr-CH', {
    day: '2-digit', month: '2-digit', year: 'numeric',
    hour: '2-digit', minute: '2-digit',
  });
};

const fmtShort = (dateStr) => {
  if (!dateStr) return '—';
  return new Date(dateStr).toLocaleString('fr-CH', {
    hour: '2-digit', minute: '2-digit',
    day: '2-digit', month: '2-digit',
  });
};

const resolveBookingStatusKey = (bookingSummary) => {
  if (!bookingSummary) return '';
  const raw = String(bookingSummary.status || '').toUpperCase();
  const normalized = raw === 'CANCELLED' ? 'CANCELED' : raw;
  const returnRaw = String(bookingSummary.return_booking?.status || '').toUpperCase();
  const returnStatus = returnRaw === 'CANCELLED' ? 'CANCELED' : returnRaw;
  const overall = String(bookingSummary.overall_status || '').toLowerCase();

  const hasReturn = Boolean(bookingSummary.return_booking);
  const returnCompleted = ['COMPLETED', 'RETURN_COMPLETED'].includes(returnStatus);
  const returnCancelled = returnStatus === 'CANCELED';
  const outboundCompleted = ['COMPLETED', 'RETURN_COMPLETED'].includes(normalized);

  if (hasReturn && overall) {
    if (overall === 'completed') return 'RETURN_COMPLETED';
    if (overall === 'cancelled') return 'CANCELED';
    if (overall === 'outbound_completed') return 'OUTBOUND_COMPLETED';
    if (overall === 'in_progress') return 'IN_PROGRESS';
    if (overall === 'planned') return 'ACCEPTED';
  }

  if (hasReturn) {
    if (returnCompleted) return 'RETURN_COMPLETED';
    if (returnCancelled) return 'CANCELED';
    if (outboundCompleted) return 'OUTBOUND_COMPLETED';
  }

  if (
    bookingSummary.completed_at &&
    !hasReturn &&
    normalized !== 'RETURN_COMPLETED' &&
    normalized !== 'CANCELED'
  ) {
    return 'COMPLETED';
  }
  return normalized;
};

// ─── Status badge ──────────────────────────────────────────
const StatusBadge = ({ request }) => {
  if (isExternalRequest(request)) {
    const info = REQUEST_STATUS_MAP[request.status] || {
      label: getRequestStatusLabel(request),
      css: 'statusExternalAssigned',
    };
    return <span className={`${s.statusBadge} ${s[info.css]}`}>{info.label}</span>;
  }
  if (hasBooking(request) && request.status === 'CONVERTED' && request.booking_summary?.status) {
    const info = BOOKING_STATUS_MAP[resolveBookingStatusKey(request.booking_summary)];
    if (info) return <span className={`${s.statusBadge} ${s[info.css]}`}>{info.label}</span>;
  }
  const info = REQUEST_STATUS_MAP[request.status] || { label: request.status, css: 'statusDraft' };
  return <span className={`${s.statusBadge} ${s[info.css]}`}>{info.label}</span>;
};

// ─── Billing Section ───────────────────────────────────────
const BillingSection = ({ request, canBilling, billingMutation, bookingBillingMutation }) => {
  const isConverted = hasBooking(request);
  const bs = isConverted ? request.booking_summary : null;
  const isInvoiced = isConverted && bs?.is_invoiced;

  const isCancelled = isConverted
    && ['CANCELED', 'CANCELLED', 'canceled', 'cancelled'].includes(bs?.status || '');
  const billable = bs?.is_cancellation_billable;

  const currentBilledTo = isConverted
    ? (bs.billed_to_type === 'clinic' ? 'institution' : bs.billed_to_type)
    : (request.billing_intent || 'patient');

  const [selectedIntent, setSelectedIntent] = useState(currentBilledTo);
  useEffect(() => {
    setSelectedIntent(currentBilledTo);
  }, [currentBilledTo]);
  const hasChanged = selectedIntent !== currentBilledTo;

  const billingLabels = {
    patient: 'Facturé au patient',
    institution: 'Facturé à l\'institution',
    clinic: 'Facturé à l\'institution',
    insurance: 'Facturé à l\'assurance',
  };

  const billingCss = {
    patient: s.billingStatusPatient,
    institution: s.billingStatusInstitution,
    clinic: s.billingStatusInstitution,
    insurance: s.billingStatusInsurance,
  };

  const handleSave = () => {
    if (!hasChanged) return;
    if (!window.confirm(
      `Modifier la facturation vers « ${selectedIntent === 'institution' ? 'Institution' : 'Patient'} » ?`
    )) return;

    if (isConverted) {
      bookingBillingMutation.mutate(
        { bookingId: request.booking_id, data: { billing_intent: selectedIntent } },
        {
          onSuccess: () => toast.success('Facturation mise à jour.'),
          onError: (err) => {
            if (err?.response?.status === 409) {
              toast.error('Transport déjà facturé. Contactez l\'entreprise de transport.');
            } else {
              toast.error(err?.response?.data?.error || 'Erreur lors de la mise à jour.');
            }
          },
        }
      );
    } else {
      billingMutation.mutate(
        { requestId: request.id, data: { billing_intent: selectedIntent } },
        {
          onSuccess: () => toast.success('Intention de facturation mise à jour.'),
          onError: (err) => toast.error(err?.response?.data?.error || 'Erreur'),
        }
      );
    }
  };

  const isPending = isConverted ? bookingBillingMutation.isPending : billingMutation.isPending;

  if (isCancelled && billable === false) {
    return (
      <div className={s.card}>
        <div className={s.cardHeader}>
          <div className={`${s.cardIcon} ${s.cardIconWarning}`}><FaFileInvoiceDollar /></div>
          <h3 className={s.cardTitle}>Facturation</h3>
        </div>
        <div className={`${s.billingStatus} ${s.billingStatusCancelled}`}>
          Annulée — non facturée
        </div>
        {bs.cancellation_display_label && (
          <p className={s.billingMuted}>Motif : {bs.cancellation_display_label}</p>
        )}
      </div>
    );
  }

  if (isCancelled && billable === true) {
    return (
      <div className={s.card}>
        <div className={s.cardHeader}>
          <div className={`${s.cardIcon} ${s.cardIconWarning}`}><FaFileInvoiceDollar /></div>
          <h3 className={s.cardTitle}>Facturation</h3>
        </div>
        <div className={`${s.billingStatus} ${s.billingStatusDanger}`}>
          Annulée — facturée
          <span className={s.billingBadgeDanger}>Facturation maintenue</span>
        </div>
        <p className={s.billingMuted}>
          {billingLabels[currentBilledTo] || currentBilledTo}
        </p>
        {bs.cancellation_display_label && (
          <p className={s.billingMuted}>Motif : {bs.cancellation_display_label}</p>
        )}
      </div>
    );
  }

  return (
    <div className={s.card}>
      <div className={s.cardHeader}>
        <div className={`${s.cardIcon} ${s.cardIconWarning}`}><FaFileInvoiceDollar /></div>
        <h3 className={s.cardTitle}>Facturation</h3>
      </div>

      <div className={`${s.billingStatus} ${billingCss[currentBilledTo] || s.billingStatusPatient}`}>
        {billingLabels[currentBilledTo] || currentBilledTo}
      </div>

      {isInvoiced && (
        <div className={s.invoicedWarning}>
          <span className={s.invoicedWarningTitle}>Transport facturé</span>
          Ce transport a été facturé par l'entreprise de transport.
          Pour modifier la facturation, contactez directement le transporteur
          afin qu'il annule la facture concernée.
        </div>
      )}

      {canBilling && !isInvoiced && (
        <div className={s.billingEdit}>
          <select
            value={selectedIntent}
            onChange={(e) => setSelectedIntent(e.target.value)}
            disabled={isPending}
            className={s.billingSelect}
          >
            <option value="patient">Patient</option>
            <option value="institution">Institution / Clinique</option>
          </select>
          <button
            onClick={handleSave}
            disabled={!hasChanged || isPending}
            className={`${s.billingSaveBtn} ${hasChanged ? s.billingSaveBtnActive : s.billingSaveBtnDisabled}`}
          >
            {isPending ? 'Enregistrement...' : 'Enregistrer'}
          </button>
        </div>
      )}
    </div>
  );
};

// ─── Main Component ────────────────────────────────────────
const InstitutionRequestDetail = () => {
  const { public_id, requestId } = useParams();
  const navigate = useNavigate();

  const { data: meData } = useInstitutionMe();
  const { data: request, isLoading, error } = useInstitutionRequest(requestId);
  const sendMutation = useSendRequest();
  const cancelMutation = useCancelRequest();
  const assignExternalMutation = useAssignExternalCarrier();
  const completeExternalMutation = useCompleteExternalMission();
  const billingMutation = useUpdateRequestBilling();
  const bookingBillingMutation = useUpdateBookingBilling();

  const institutionRole = meData?.institution_role;
  const canManage = canManageRequests(institutionRole);
  const canBillingEdit = canEditBilling(institutionRole);
  const institutionSocket = useMemo(() => getInstitutionSocket(), []);

  const [showSendModal, setShowSendModal] = useState(false);
  const [showAssignExternalForm, setShowAssignExternalForm] = useState(false);
  const [showCompleteExternalForm, setShowCompleteExternalForm] = useState(false);
  const [externalCarrierForm, setExternalCarrierForm] = useState(EMPTY_EXTERNAL_CARRIER_FORM);
  const [externalCompleteNotes, setExternalCompleteNotes] = useState('');
  const [composingEmail, setComposingEmail] = useState(false);

  const goBack = () => navigate(`/dashboard/institution/${public_id}/requests`);

  const handleComposeCarrierEmail = useCallback(async () => {
    const email = request?.external_carrier?.email;
    if (!email || !request?.id) return;
    setComposingEmail(true);
    try {
      await exportRequestMissionPdf(request.id, { variant: 'operational' });
      toast.success('Bon téléchargé — joignez-le à l\'e-mail');
    } catch (err) {
      toast.error(err?.message || 'Erreur lors de l\'export du bon');
    } finally {
      setComposingEmail(false);
    }
    const institutionName = meData?.institution?.name || meData?.name || '';
    window.location.href = buildCarrierMailto(email, request, {
      institutionName,
      institutionPhone: meData?.contact_phone,
    });
  }, [request, meData]);

  const handleAssignExternalCarrier = async () => {
    const validationError = validateExternalCarrierForm(externalCarrierForm);
    if (validationError) {
      toast.error(validationError);
      return;
    }
    try {
      await assignExternalMutation.mutateAsync({
        requestId: request.id,
        data: buildExternalCarrierPayload(externalCarrierForm),
      });
      setShowAssignExternalForm(false);
      setExternalCarrierForm(EMPTY_EXTERNAL_CARRIER_FORM);
      toast.success('Transporteur externe affecté');
    } catch (err) {
      toast.error(err?.response?.data?.error || 'Erreur lors de l\'affectation externe');
    }
  };

  const handleCompleteExternalMission = async () => {
    try {
      await completeExternalMutation.mutateAsync({
        requestId: request.id,
        data: {
          executed_at: new Date().toISOString(),
          notes: externalCompleteNotes.trim() || undefined,
        },
      });
      setShowCompleteExternalForm(false);
      setExternalCompleteNotes('');
      toast.success('Mission déclarée réalisée par l\'institution');
    } catch (err) {
      toast.error(err?.response?.data?.error || 'Erreur lors de la déclaration');
    }
  };

  const handleSend = useCallback(async () => {
    try {
      await sendMutation.mutateAsync({ requestId: request.id, options: {} });
      setShowSendModal(false);
      toast.success('Demande envoyée');
    } catch (err) {
      toast.error(err?.response?.data?.error || 'Erreur lors de l\'envoi');
    }
  }, [sendMutation, request?.id]);

  const handleCancel = async () => {
    try {
      await cancelMutation.mutateAsync({ requestId: request.id, reason: '' });
      toast.success('Demande annulée');
    } catch (err) {
      if (err?.response?.status === 409) {
        toast.error(`Demande convertie en booking. Annulez le booking #${err.response.data.resulting_booking_id} directement.`);
      } else {
        toast.error(err?.response?.data?.error || 'Erreur lors de l\'annulation');
      }
    }
  };

  // Timeline
  const getTimeline = () => {
    const events = [];
    const creator = request?.created_by_name;
    const company = request?.accepted_by_company?.name;
    const bs = request?.booking_summary;

    if (request?.created_at)
      events.push({ event: `Demande créée${creator ? ` par ${creator}` : ''}`, date: request.created_at });
    if (request?.sent_at)
      events.push({ event: 'Envoyée aux transporteurs', date: request.sent_at });
    if (request?.accepted_at)
      events.push({ event: `Acceptée${company ? ` par ${company}` : ''}`, date: request.accepted_at });
    if (request?.converted_at)
      events.push({ event: 'Convertie en booking', date: request.converted_at });

    // Historique opérationnel consolidé du transport (legs multi-étapes +
    // retours) si disponible, sinon le booking unique du résumé.
    const journey = Array.isArray(bs?.route_journey) ? bs.route_journey : null;
    if (journey && journey.length) {
      journey.forEach((ev) => {
        if (ev?.date) events.push({ event: ev.event, date: ev.date, type: ev.type });
      });
    } else {
      if (bs?.boarded_at)
        events.push({ event: 'Patient pris en charge', date: bs.boarded_at });
      if (bs?.completed_at)
        events.push({ event: 'Transport terminé', date: bs.completed_at });
    }
    const bsCancelled = bs?.cancelled_at;
    if (bsCancelled) {
      const roleMap = { company: 'Entreprise', driver: 'Chauffeur', admin: 'Admin', system: 'Système' };
      const byLabel = roleMap[bs.cancelled_by_role] || '';
      const reasonLabel = bs.cancellation_display_label || '';
      const billableFlag = bs.is_cancellation_billable;

      let detail = 'Annulée';
      if (byLabel) detail += ` par ${byLabel}`;
      if (reasonLabel) detail += ` — ${reasonLabel}`;
      if (billableFlag === true) detail += ' (facturée)';
      else if (billableFlag === false) detail += ' (non facturée)';

      events.push({ event: detail, date: bsCancelled, type: 'cancel' });
    } else if (request?.cancelled_at) {
      events.push({ event: 'Annulée', date: request.cancelled_at, type: 'cancel' });
    }

    if (isExternalRequest(request)) {
      const ext = request.external_carrier || {};
      if (ext.assigned_at) {
        events.push({
          event: `Transporteur externe affecté${ext.name ? ` — ${ext.name}` : ''}`,
          date: ext.assigned_at,
        });
      }
      if (ext.executed_at) {
        events.push({
          event: 'Déclarée réalisée par l\'institution',
          date: ext.executed_at,
        });
      }
    }

    return events.sort((a, b) => new Date(b.date) - new Date(a.date));
  };

  // ── Loading / Error ──
  if (isLoading) return <div className={s.loading}>Chargement...</div>;
  if (error || !request) return (
    <div className={s.errorState}>
      <p>Demande non trouvée</p>
      <button onClick={goBack}>Retour à la liste</button>
    </div>
  );

  const isExternal = isExternalRequest(request);
  const isConverted = hasBooking(request);
  const bs = isConverted ? request.booking_summary : null;
  const showAssignExternalAction = canManage && canAssignExternalCarrier(request);
  const showCompleteExternalAction = canManage && canCompleteExternalMission(request);
  const canCancelRequest = canManage && (
    ['DRAFT', 'SENT', 'ACCEPTED', EXTERNAL_STATUSES.ASSIGNED].includes(request.status)
  );

  return (
    <div className={s.page}>

      {/* ═══ HEADER ═══ */}
      <div className={s.header}>
        <div className={s.headerLeft}>
          <button className={s.backLink} onClick={goBack}>
            <FaArrowLeft size={10} /> Retour aux demandes
          </button>
          <div className={s.headerTitle}>
            <h1 className={s.requestId}>Demande #{request.id}</h1>
            <StatusBadge request={request} />
          </div>
        </div>

        <div className={s.headerActions}>
          {canManage && request.status === 'DRAFT' && !isExternal && (
            <>
              <button
                className={`${s.actionBtn} ${s.btnSecondary}`}
                onClick={() => navigate(`/dashboard/institution/${public_id}/requests/${request.id}?edit=true`)}
              >
                <FaEdit size={12} /> Modifier
              </button>
              <button
                className={`${s.actionBtn} ${s.btnPrimary}`}
                onClick={() => setShowSendModal(true)}
                disabled={sendMutation.isPending}
              >
                <FaPaperPlane size={12} /> Envoyer
              </button>
            </>
          )}
          {showAssignExternalAction && !showAssignExternalForm && (
            <button
              type="button"
              className={`${s.actionBtn} ${s.btnSecondary}`}
              onClick={() => setShowAssignExternalForm(true)}
            >
              <FaTruck size={12} /> Affecter transporteur externe
            </button>
          )}
          {showCompleteExternalAction && !showCompleteExternalForm && (
            <button
              type="button"
              className={`${s.actionBtn} ${s.btnPrimary}`}
              onClick={() => setShowCompleteExternalForm(true)}
            >
              <FaTruck size={12} /> Déclarer réalisée
            </button>
          )}
          {canCancelRequest && (
            <button
              className={`${s.actionBtn} ${s.btnDanger}`}
              onClick={handleCancel}
              disabled={cancelMutation.isPending}
            >
              <FaTimes size={12} /> Annuler
            </button>
          )}
        </div>
      </div>

      {showAssignExternalAction && showAssignExternalForm && (
        <div className={s.card}>
          <ExternalCarrierFields
            value={externalCarrierForm}
            onChange={setExternalCarrierForm}
            idPrefix={`detail-assign-external-${request.id}`}
          />
          <div className={s.headerActions}>
            <button
              type="button"
              className={`${s.actionBtn} ${s.btnSecondary}`}
              onClick={() => {
                setShowAssignExternalForm(false);
                setExternalCarrierForm(EMPTY_EXTERNAL_CARRIER_FORM);
              }}
            >
              Annuler
            </button>
            <button
              type="button"
              className={`${s.actionBtn} ${s.btnPrimary}`}
              onClick={handleAssignExternalCarrier}
              disabled={assignExternalMutation.isPending}
            >
              Confirmer l&apos;affectation
            </button>
          </div>
        </div>
      )}

      {showCompleteExternalAction && showCompleteExternalForm && (
        <div className={s.card}>
          <p className={s.billingMuted}>
            Déclaration manuelle : la mission sera marquée comme réalisée par l&apos;institution.
          </p>
          <textarea
            rows={3}
            value={externalCompleteNotes}
            onChange={(e) => setExternalCompleteNotes(e.target.value)}
            placeholder="Notes (optionnel)"
            className={s.billingSelect}
          />
          <div className={s.headerActions}>
            <button
              type="button"
              className={`${s.actionBtn} ${s.btnSecondary}`}
              onClick={() => {
                setShowCompleteExternalForm(false);
                setExternalCompleteNotes('');
              }}
            >
              Annuler
            </button>
            <button
              type="button"
              className={`${s.actionBtn} ${s.btnPrimary}`}
              onClick={handleCompleteExternalMission}
              disabled={completeExternalMutation.isPending}
            >
              Confirmer la déclaration
            </button>
          </div>
        </div>
      )}

      {/* ═══ CARD 1 — Transport summary (hero) ═══ */}
      {isConverted && !isExternal && bs && (
        <div className={s.summaryCard}>
          <div className={s.summaryTop}>
            <div className={s.summaryTopIcon}><FaTruck /></div>
            <div className={s.summaryTopInfo}>
              <p className={s.summaryCompany}>
                {request.accepted_by_company?.name || 'Transporteur'}
              </p>
              <p className={s.summaryMeta}>
                Booking #{request.booking_id}
              </p>
            </div>
          </div>
          <div className={s.summaryBody}>
            <div className={s.summaryGrid}>
              <div className={s.summaryItem}>
                <span className={s.summaryLabel}>Horaire prévu</span>
                <span className={s.summaryValue}>
                  {bs.scheduled_time
                    ? new Date(bs.scheduled_time).toLocaleString('fr-CH', {
                        day: '2-digit', month: '2-digit', year: 'numeric',
                        hour: '2-digit', minute: '2-digit',
                      })
                    : '—'}
                </span>
              </div>
              <div className={s.summaryItem}>
                <span className={s.summaryLabel}>Patient</span>
                <span className={s.summaryValue}>{bs.customer_name || '—'}</span>
              </div>
              <div className={s.summaryItem}>
                <span className={s.summaryLabel}>Statut du transport</span>
                <span className={s.summaryValue}>
                  {BOOKING_STATUS_MAP[resolveBookingStatusKey(bs)]?.label || 'En cours'}
                </span>
              </div>
              {bs.amount != null && (
                <div className={s.summaryItem}>
                  <span className={s.summaryLabel}>Montant</span>
                  <span className={s.summaryValue}>{Number(bs.amount).toFixed(2)} CHF</span>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {isExternal && (
        <div className={s.summaryCard}>
          <div className={s.summaryTop}>
            <div className={s.summaryTopIcon}><FaTruck /></div>
            <div className={s.summaryTopInfo}>
              <p className={s.summaryCompany}>Transporteur externe</p>
              <p className={s.summaryMeta}>{request.external_carrier?.name || '—'}</p>
            </div>
          </div>
          <div className={s.summaryBody}>
            <div className={s.summaryGrid}>
              {request.external_carrier?.phone && (
                <div className={s.summaryItem}>
                  <span className={s.summaryLabel}>Téléphone</span>
                  <span className={s.summaryValue}>{request.external_carrier.phone}</span>
                </div>
              )}
              {request.external_carrier?.email && (
                <div className={s.summaryItem}>
                  <span className={s.summaryLabel}>Email</span>
                  <span className={s.summaryValue}>
                    <a href={`mailto:${request.external_carrier.email}`}>{request.external_carrier.email}</a>
                  </span>
                </div>
              )}
              {request.external_carrier?.reference && (
                <div className={s.summaryItem}>
                  <span className={s.summaryLabel}>Référence</span>
                  <span className={s.summaryValue}>{request.external_carrier.reference}</span>
                </div>
              )}
              {request.external_carrier?.reason && (
                <div className={s.summaryItem}>
                  <span className={s.summaryLabel}>Raison</span>
                  <span className={s.summaryValue}>{request.external_carrier.reason}</span>
                </div>
              )}
              {request.external_carrier?.executed_at && (
                <div className={s.summaryItem}>
                  <span className={s.summaryLabel}>Déclarée réalisée le</span>
                  <span className={s.summaryValue}>{fmt(request.external_carrier.executed_at)}</span>
                </div>
              )}
            </div>
            {request.external_carrier?.email && (
              <div className={s.carrierEmailActions}>
                <button
                  type="button"
                  className={s.carrierEmailBtn}
                  onClick={handleComposeCarrierEmail}
                  disabled={composingEmail}
                  title="Télécharge le bon de transport et ouvre un e-mail pré-rempli pour le transporteur"
                >
                  <FaEnvelope size={12} />
                  {composingEmail ? 'Préparation…' : 'Ouvrir dans ma messagerie'}
                </button>
                <span className={s.carrierEmailHint}>
                  Le bon (PDF) est téléchargé : joignez-le à l&apos;e-mail avant l&apos;envoi.
                </span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ═══ CARD 2 — Route ═══ */}
      <div className={s.card}>
        <div className={s.cardHeader}>
          <div className={`${s.cardIcon} ${s.cardIconBrand}`}><FaRoute /></div>
          <h3 className={s.cardTitle}>Trajet</h3>
        </div>

        <div className={s.route}>
          <div className={s.routeTrack}>
            <div className={`${s.routeDot} ${s.routeDotStart}`} />
            <div className={s.routeLine} />
            <div className={`${s.routeDot} ${s.routeDotEnd}`} />
          </div>
          <div className={s.routeStops}>
            <div className={s.routeStop}>
              <div className={s.routeStopLabel}>Départ</div>
              <div className={s.routeStopAddress}>{request.pickup_location || '—'}</div>
            </div>
            <div className={s.routeStop}>
              <div className={s.routeStopLabel}>Arrivée</div>
              <div className={s.routeStopAddress}>{request.dropoff_location || '—'}</div>
            </div>
          </div>
        </div>

        {request.round_trip && (
          <div className={s.roundTripBadge}>Aller-retour</div>
        )}
      </div>

      {/* ═══ CARD 4 — Détails ═══ */}
      <div className={s.card}>
        <div className={s.cardHeader}>
          <div className={`${s.cardIcon} ${s.cardIconBlue}`}><FaInfoCircle /></div>
          <h3 className={s.cardTitle}>Détails</h3>
        </div>

        <div className={s.infoRow}>
          <span className={s.infoLabel}>Type</span>
          <span className={s.infoValue}>{MISSION_LABELS[request.mission_type] || request.mission_type}</span>
        </div>
        <div className={s.infoRow}>
          <span className={s.infoLabel}>Date et heure</span>
          <span className={s.infoValue}>{fmt(request.scheduled_time)}</span>
        </div>
        {request.round_trip && request.return_time && (
          <div className={s.infoRow}>
            <span className={s.infoLabel}>Retour</span>
            <span className={s.infoValue}>{fmt(request.return_time)}</span>
          </div>
        )}
        {request.external_reference && (
          <div className={s.infoRow}>
            <span className={s.infoLabel}>Réf. externe</span>
            <span className={`${s.infoValue} ${s.infoMono}`}>{request.external_reference}</span>
          </div>
        )}
        <div className={s.infoRow}>
          <span className={s.infoLabel}>ID public</span>
          <span className={`${s.infoValue} ${s.infoMono}`}>{request.public_id}</span>
        </div>
      </div>

      {/* ═══ CARD 5 — Besoins spécifiques (uniquement si au moins un actif ou notes) ═══ */}
      {(request.requires_wheelchair || request.requires_stretcher || request.requires_oxygen || request.notes) && (
        <div className={s.card}>
          <div className={s.cardHeader}>
            <div className={`${s.cardIcon} ${s.cardIconMuted}`}><FaNotesMedical /></div>
            <h3 className={s.cardTitle}>Besoins spécifiques</h3>
          </div>

          <div className={s.needsRow}>
            {request.requires_wheelchair && (
              <span className={`${s.needsChip} ${s.needsChipActive}`}>
                <FaWheelchair size={11} /> Fauteuil roulant
              </span>
            )}
            {request.requires_stretcher && (
              <span className={`${s.needsChip} ${s.needsChipActive}`}>
                Brancard
              </span>
            )}
            {request.requires_oxygen && (
              <span className={`${s.needsChip} ${s.needsChipDanger}`}>
                O₂ Oxygène
              </span>
            )}
          </div>

          {request.notes && (
            <div className={s.notesBlock} style={{ marginTop: 14 }}>
              {request.notes}
            </div>
          )}
        </div>
      )}

      {/* ═══ CARD 6 — Facturation ═══ */}
      {(canBillingEdit || request.billing_intent) && (
        <BillingSection
          key={request.id}
          request={request}
          canBilling={canBillingEdit}
          billingMutation={billingMutation}
          bookingBillingMutation={bookingBillingMutation}
        />
      )}

      {/* ═══ Communication (si booking converti et chauffeur assigné ou en course) ═══ */}
      {isConverted && !isExternal && request.booking_id && (() => {
        const chatActiveStatuses = ['ASSIGNED', 'EN_ROUTE', 'IN_PROGRESS'];
        const statusNorm = resolveBookingStatusKey(bs);
        const canShowChat = chatActiveStatuses.includes(statusNorm);
        if (!canShowChat) {
          return (
            <div className={s.card}>
              <div className={s.cardHeader}>
                <div className={`${s.cardIcon} ${s.cardIconMuted}`}>💬</div>
                <h3 className={s.cardTitle}>Communication</h3>
              </div>
              <p className={s.infoValue} style={{ margin: 0 }}>
                Le transport a été confirmé. Le chauffeur sera assigné prochainement.
              </p>
            </div>
          );
        }
        return (
          <BookingChat
            bookingId={request.booking_id}
            socket={institutionSocket}
            fetchMessages={fetchBookingMessages}
            sendMessage={sendBookingMessage}
            closed={['COMPLETED', 'RETURN_COMPLETED', 'CANCELED'].includes(resolveBookingStatusKey(bs))}
          />
        );
      })()}

      {/* ═══ CARD 7 — Historique ═══ */}
      <div className={s.card}>
        <div className={s.cardHeader}>
          <div className={`${s.cardIcon} ${s.cardIconMuted}`}><FaHistory /></div>
          <h3 className={s.cardTitle}>Historique</h3>
        </div>

        <div className={s.timeline}>
          {getTimeline().map((item, i) => (
            <div key={i} className={`${s.timelineItem} ${item.type === 'cancel' ? s.timelineItemCancel : ''}`}>
              <div className={s.timelineEvent}>{item.event}</div>
              <div className={s.timelineDate}>{fmtShort(item.date)}</div>
            </div>
          ))}
        </div>
      </div>

      {showSendModal && (
        <ConfirmSendModal
          onClose={() => setShowSendModal(false)}
          onConfirm={handleSend}
          loading={sendMutation.isPending}
        />
      )}
    </div>
  );
};

export default InstitutionRequestDetail;
