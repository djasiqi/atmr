// pages/institution/Requests/InstitutionRequestDetail.jsx
/**
 * Fiche détaillée d'un transport — design professionnel SaaS B2B
 */

import React, { useState, useMemo } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  FaArrowLeft, FaEdit, FaPaperPlane, FaTimes,
  FaTruck, FaRoute, FaFileInvoiceDollar,
  FaHistory, FaWheelchair, FaInfoCircle, FaNotesMedical,
} from 'react-icons/fa';
import {
  useInstitutionRequest, useInstitutionMe,
  useSendRequest, useCancelRequest,
  useUpdateRequestBilling, useUpdateBookingBilling,
} from '../../../hooks/useInstitutionData';
import { canManageRequests, canEditBilling } from '../../../utils/institutionPermissions';
import { toast } from 'sonner';
import { getInstitutionSocket } from '../../../services/institutionSocket';
import { fetchBookingMessages, sendBookingMessage } from '../../../services/institutionService';
import BookingChat from '../../company/Reservations/components/BookingChat';
import s from './InstitutionRequestDetail.module.css';

// ─── Status mapping ────────────────────────────────────────
const BOOKING_STATUS_MAP = {
  PENDING:          { label: 'En attente',           css: 'statusPending' },
  ACCEPTED:         { label: 'Accepté',              css: 'statusAccepted' },
  ASSIGNED:         { label: 'Chauffeur assigné',    css: 'statusAssigned' },
  EN_ROUTE:         { label: 'En route',             css: 'statusEnRoute' },
  IN_PROGRESS:      { label: 'En cours',             css: 'statusInProgress' },
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

// ─── Status badge ──────────────────────────────────────────
const StatusBadge = ({ request }) => {
  if (request.status === 'CONVERTED' && request.booking_summary?.status) {
    const info = BOOKING_STATUS_MAP[request.booking_summary.status];
    if (info) return <span className={`${s.statusBadge} ${s[info.css]}`}>{info.label}</span>;
  }
  const info = REQUEST_STATUS_MAP[request.status] || { label: request.status, css: 'statusDraft' };
  return <span className={`${s.statusBadge} ${s[info.css]}`}>{info.label}</span>;
};

// ─── Billing Section ───────────────────────────────────────
const BillingSection = ({ request, canBilling, billingMutation, bookingBillingMutation }) => {
  const isConverted = request.status === 'CONVERTED' && request.booking_summary;
  const bs = request.booking_summary;
  const isInvoiced = isConverted && bs?.is_invoiced;

  const isCancelled = isConverted
    && ['CANCELED', 'CANCELLED', 'canceled', 'cancelled'].includes(bs?.status || '');
  const billable = bs?.is_cancellation_billable;

  const currentBilledTo = isConverted
    ? (bs.billed_to_type === 'clinic' ? 'institution' : bs.billed_to_type)
    : (request.billing_intent || 'patient');

  const [selectedIntent, setSelectedIntent] = useState(currentBilledTo);
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
  const billingMutation = useUpdateRequestBilling();
  const bookingBillingMutation = useUpdateBookingBilling();

  const institutionRole = meData?.institution_role;
  const canManage = canManageRequests(institutionRole);
  const canBillingEdit = canEditBilling(institutionRole);
  const institutionSocket = useMemo(() => getInstitutionSocket(), []);

  const goBack = () => navigate(`/dashboard/institution/${public_id}/requests`);

  const handleSend = async () => {
    if (!window.confirm('Envoyer cette demande aux transporteurs ?')) return;
    try {
      await sendMutation.mutateAsync({ requestId: request.id, options: {} });
      toast.success('Demande envoyée');
    } catch (err) {
      toast.error(err?.response?.data?.error || 'Erreur lors de l\'envoi');
    }
  };

  const handleCancel = async () => {
    const reason = window.prompt('Raison de l\'annulation (optionnel) :');
    if (reason === null) return;
    try {
      await cancelMutation.mutateAsync({ requestId: request.id, reason });
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
    if (bs?.boarded_at)
      events.push({ event: 'Patient pris en charge', date: bs.boarded_at });
    if (bs?.completed_at)
      events.push({ event: 'Transport terminé', date: bs.completed_at });
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

  const isConverted = request.status === 'CONVERTED' && request.booking_id;
  const bs = request.booking_summary;

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
          {canManage && request.status === 'DRAFT' && (
            <>
              <button
                className={`${s.actionBtn} ${s.btnSecondary}`}
                onClick={() => navigate(`/dashboard/institution/${public_id}/requests/${request.id}?edit=true`)}
              >
                <FaEdit size={12} /> Modifier
              </button>
              <button
                className={`${s.actionBtn} ${s.btnPrimary}`}
                onClick={handleSend}
                disabled={sendMutation.isPending}
              >
                <FaPaperPlane size={12} /> Envoyer
              </button>
            </>
          )}
          {canManage && ['DRAFT', 'SENT', 'ACCEPTED'].includes(request.status) && (
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

      {/* ═══ CARD 1 — Transport summary (hero) ═══ */}
      {isConverted && bs && (
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
                  {BOOKING_STATUS_MAP[bs.status]?.label || 'En cours'}
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
          request={request}
          canBilling={canBillingEdit}
          billingMutation={billingMutation}
          bookingBillingMutation={bookingBillingMutation}
        />
      )}

      {/* ═══ Communication (si booking converti) ═══ */}
      {isConverted && request.booking_id && (
        <BookingChat
          bookingId={request.booking_id}
          socket={institutionSocket}
          fetchMessages={fetchBookingMessages}
          sendMessage={sendBookingMessage}
          closed={['COMPLETED', 'RETURN_COMPLETED', 'CANCELED', 'CANCELLED'].includes(bs?.status || '')}
        />
      )}

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
    </div>
  );
};

export default InstitutionRequestDetail;
