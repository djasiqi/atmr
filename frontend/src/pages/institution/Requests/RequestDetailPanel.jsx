// pages/institution/Requests/RequestDetailPanel.jsx
/**
 * Panel latéral du détail d'une demande — intégré dans la page liste.
 * Réutilise la même logique que InstitutionRequestDetail mais dans un format panel.
 */

import React, { useState, useCallback, useMemo, useEffect, useRef } from 'react';
import ConfirmSendModal from './ConfirmSendModal';
import ChipSelect from '../../../components/ui/ChipSelect';
import {
  FaTimes, FaEdit, FaPaperPlane,
  FaTruck, FaRoute, FaFileInvoiceDollar, FaFilePdf,
  FaHistory, FaWheelchair, FaInfoCircle, FaNotesMedical,
  FaPhoneAlt, FaEnvelope,
} from 'react-icons/fa';
import { HiOutlineX } from 'react-icons/hi';
import {
  useInstitutionRequest, useInstitutionMe,
  useSendRequest, useCancelRequest,
  useUpdateRequestBilling, useUpdateBookingBilling,
  usePatchInstitutionBooking, useCancelInstitutionBooking,
  useRequestTimeline, useReleaseBookingForRedispatch,
  useAssignExternalCarrier, useCompleteExternalMission,
  institutionQueryKeys,
} from '../../../hooks/useInstitutionData';
import { useQueryClient } from '@tanstack/react-query';
import {
  canManageRequests,
  canEditBilling,
  canViewFinancialAmounts,
  canViewBillingSection,
  canExportTransports,
} from '../../../utils/institutionPermissions';
import InstitutionOperationalEdit from './InstitutionOperationalEdit';
import { formatLegTime, formatReturnTimeLabel, formatDepartureTime } from '../../../utils/formatLegTime';
import InstitutionRequestEdit from './InstitutionRequestEdit';
import { getAuthEnv } from '../../../utils/webAuthSession';
import { toast } from 'sonner';
import { getInstitutionSocket } from '../../../services/institutionSocket';
import { fetchBookingMessages, sendBookingMessage, exportRequestMissionPdf } from '../../../services/institutionService';
import { buildCarrierMailto } from '../../../utils/externalCarrierEmail';
import BookingChat from '../../company/Reservations/components/BookingChat';
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
  EXTERNAL_STATUSES,
} from '../../../utils/requestStatus';
import { BOOKING_STATUS_LABELS } from './statusColors';
import s from './RequestDetailPanel.module.css';

const MISSION_LABELS = {
  patient_transport: 'Transport patient',
  material_delivery: 'Livraison matériel',
};
const DEMO_COMPANY_NAME = 'LIRIE Transport Démo';
const DEMO_INSTITUTION_SESSION_KEY = 'demo_institution_request_simulation_state';
const DEMO_INSTITUTION_COMPLETED_KEY = 'demo_institution_journey_completed';

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

const getRoutePoints = (request) => {
  const legs = Array.isArray(request?.legs)
    ? [...request.legs].sort((a, b) => (a.sequence_index ?? 0) - (b.sequence_index ?? 0))
    : [];
  if (legs.length > 0) {
    return [
      { label: 'Départ', address: legs[0].pickup_location, kind: 'start' },
      ...legs.map((leg, index) => {
        const isReturn = Boolean(request?.return_to_institution) && index === legs.length - 1;
        const timeLabel = index === 0
          ? formatDepartureTime(request)
          : formatLegTime(leg);
        return {
          label: isReturn ? 'Retour' : `Destination ${index + 1}`,
          address: leg.dropoff_location,
          kind: isReturn ? 'return' : 'destination',
          timeLabel,
          details: {
            establishment: leg.dropoff_establishment,
            service: leg.dropoff_service,
            doctor: leg.dropoff_doctor,
          },
        };
      }),
    ];
  }
  return [
    { label: 'Départ', address: request?.pickup_location, kind: 'start' },
    { label: 'Destination 1', address: request?.dropoff_location, kind: 'destination' },
  ];
};

const getTripBadge = (request, routePoints) => {
  if (request?.return_to_institution) {
    return {
      className: 'roundTripBadge',
      label: `A/R institution — ${Math.max(routePoints.length - 1, 1)} trajet(s)`,
    };
  }
  if (request?.multi_stop || routePoints.length > 2) {
    return {
      className: 'multiStopBadge',
      label: `${routePoints.length - 1} destination(s)`,
    };
  }
  if (request?.is_round_trip || request?.round_trip) {
    const returnHint = formatReturnTimeLabel(request);
    return {
      className: 'roundTripBadge',
      label: `Aller-retour${returnHint ? ` — ${returnHint}` : ''}`,
    };
  }
  return {
    className: 'oneWayBadge',
    label: 'Aller simple',
  };
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
  if (
    bookingSummary.boarded_at &&
    normalized !== 'COMPLETED' &&
    normalized !== 'RETURN_COMPLETED' &&
    normalized !== 'CANCELED'
  ) {
    return 'IN_PROGRESS';
  }
  return normalized;
};

const ExternalCarrierSection = ({ request, onComposeEmail, composing = false }) => {
  const ext = request?.external_carrier || {};
  const phone = (ext.phone || '').trim();
  const phoneHref = phone ? `tel:${phone.replace(/[^+0-9]/g, '')}` : '';
  const email = (ext.email || '').trim();

  return (
    <div className={s.section}>
      <div className={s.sectionHeader}>
        <div className={`${s.sectionIcon} ${s.sectionIconBrand}`}><FaTruck /></div>
        <h3 className={s.sectionTitle}>Transporteur externe</h3>
      </div>
      <div className={s.summaryGrid}>
        <div className={s.summaryItem}>
          <span className={s.summaryLabel}>Nom</span>
          <span className={s.summaryValue}>{ext.name || '—'}</span>
        </div>
        {phone && (
          <div className={s.summaryItem}>
            <span className={s.summaryLabel}>Téléphone</span>
            <span className={s.summaryValue}>
              <a href={phoneHref} className={s.carrierContactItem}>{phone}</a>
            </span>
          </div>
        )}
        {ext.email && (
          <div className={s.summaryItem}>
            <span className={s.summaryLabel}>Email</span>
            <span className={s.summaryValue}>
              <a href={`mailto:${ext.email}`} className={s.carrierContactItem}>{ext.email}</a>
            </span>
          </div>
        )}
        {ext.reference && (
          <div className={s.summaryItem}>
            <span className={s.summaryLabel}>Référence</span>
            <span className={s.summaryValue}>{ext.reference}</span>
          </div>
        )}
        {ext.reason && (
          <div className={s.summaryItem}>
            <span className={s.summaryLabel}>Raison</span>
            <span className={s.summaryValue}>{ext.reason}</span>
          </div>
        )}
        {ext.assigned_at && (
          <div className={s.summaryItem}>
            <span className={s.summaryLabel}>Affecté le</span>
            <span className={s.summaryValue}>{fmt(ext.assigned_at)}</span>
          </div>
        )}
        {ext.externalized_by_name && (
          <div className={s.summaryItem}>
            <span className={s.summaryLabel}>Affecté par</span>
            <span className={s.summaryValue}>{ext.externalized_by_name}</span>
          </div>
        )}
        {ext.executed_at && (
          <div className={s.summaryItem}>
            <span className={s.summaryLabel}>Déclarée réalisée le</span>
            <span className={s.summaryValue}>{fmt(ext.executed_at)}</span>
          </div>
        )}
        {ext.executed_by_name && (
          <div className={s.summaryItem}>
            <span className={s.summaryLabel}>Déclarée par</span>
            <span className={s.summaryValue}>{ext.executed_by_name}</span>
          </div>
        )}
        {ext.execution_notes && (
          <div className={s.summaryItem}>
            <span className={s.summaryLabel}>Notes</span>
            <span className={s.summaryValue}>{ext.execution_notes}</span>
          </div>
        )}
      </div>
      {email && onComposeEmail && (
        <div className={s.carrierEmailActions}>
          <button
            type="button"
            className={s.carrierEmailBtn}
            onClick={onComposeEmail}
            disabled={composing}
            title="Télécharge le bon de transport et ouvre un e-mail pré-rempli pour le transporteur"
          >
            <FaEnvelope size={12} />
            {composing ? 'Préparation…' : 'Ouvrir dans ma messagerie'}
          </button>
          <span className={s.carrierEmailHint}>
            Le bon (PDF) est téléchargé : joignez-le à l&apos;e-mail avant l&apos;envoi.
          </span>
        </div>
      )}
    </div>
  );
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

  const [overrideReason, setOverrideReason] = useState('');

  const handleSave = () => {
    if (!hasChanged) return;
    if (!overrideReason.trim() || overrideReason.trim().length < 3) {
      toast.error('Motif obligatoire pour modifier la facturation.');
      return;
    }

    const billingPayload = {
      billing_intent: selectedIntent,
      override_reason: overrideReason.trim(),
      billing_change_reason_code: 'ADMIN_CORRECTION',
    };

    if (isConverted) {
      bookingBillingMutation.mutate(
        {
          bookingId: request.booking_summary?.id || request.booking_id,
          data: {
            ...billingPayload,
            version: request.booking_summary?.edit_version,
          },
        },
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
        { requestId: request.id, data: billingPayload },
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
      <div className={s.section}>
        <div className={s.sectionHeader}>
          <div className={`${s.sectionIcon} ${s.sectionIconWarning}`}><FaFileInvoiceDollar /></div>
          <h3 className={s.sectionTitle}>Facturation</h3>
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
      <div className={s.section}>
        <div className={s.sectionHeader}>
          <div className={`${s.sectionIcon} ${s.sectionIconWarning}`}><FaFileInvoiceDollar /></div>
          <h3 className={s.sectionTitle}>Facturation</h3>
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
    <div className={s.section}>
      <div className={s.sectionHeader}>
        <div className={`${s.sectionIcon} ${s.sectionIconWarning}`}><FaFileInvoiceDollar /></div>
        <h3 className={s.sectionTitle}>Facturation</h3>
      </div>

      <div className={`${s.billingStatus} ${billingCss[currentBilledTo] || s.billingStatusPatient}`}>
        {billingLabels[currentBilledTo] || currentBilledTo}
      </div>

      {isInvoiced && (
        <div className={s.invoicedWarning}>
          <span className={s.invoicedWarningTitle}>Transport facturé</span>
          Contactez le transporteur pour modifier la facturation.
        </div>
      )}

      {canBilling && !isInvoiced && (
        <div className={s.billingEdit}>
          <input
            className={s.editInput}
            placeholder="Motif obligatoire"
            value={overrideReason}
            onChange={(e) => setOverrideReason(e.target.value)}
          />
          <ChipSelect
            options={[
              { value: 'patient', label: 'Patient' },
              { value: 'institution', label: 'Institution' },
            ]}
            value={selectedIntent}
            onChange={(val) => setSelectedIntent(val)}
            disabled={isPending}
          />
          <button
            onClick={handleSave}
            disabled={!hasChanged || isPending}
            className={`${s.billingSaveBtn} ${hasChanged ? s.billingSaveBtnActive : ''}`}
          >
            {isPending ? '...' : 'Enregistrer'}
          </button>
        </div>
      )}
    </div>
  );
};

// ─── Main Component ────────────────────────────────────────
const RequestDetailPanel = ({ requestId, onClose }) => {
  const queryClient = useQueryClient();
  const { data: meData } = useInstitutionMe();
  const { data: request, isLoading, error } = useInstitutionRequest(requestId);
  const sendMutation = useSendRequest();
  const cancelMutation = useCancelRequest();
  const assignExternalMutation = useAssignExternalCarrier();
  const completeExternalMutation = useCompleteExternalMission();
  const billingMutation = useUpdateRequestBilling();
  const bookingBillingMutation = useUpdateBookingBilling();
  const patchBookingMutation = usePatchInstitutionBooking();
  const cancelBookingMutation = useCancelInstitutionBooking();
  const redispatchMutation = useReleaseBookingForRedispatch();
  const { data: timelineData, isLoading: timelineLoading } = useRequestTimeline(
    requestId,
    Boolean(requestId)
  );

  const timeline = useMemo(() => {
    const events = [];
    const pushEvent = (item) => {
      if (!item?.date) return;
      const key = `${item.event || ''}|${item.date}`;
      if (events.some((ev) => `${ev.event || ''}|${ev.date}` === key)) return;
      events.push(item);
    };

    const bs = request?.booking_summary;

    // Source canonique : la timeline API (libellés riches : « Offre acceptée »,
    // « Course créée », etc.). On l'utilise telle quelle si elle existe.
    const apiEvents = timelineData?.events || [];
    const hasApiTimeline = apiEvents.length > 0;
    // `request_converted` (« Réservation créée ») et `booking_created`
    // (« Course créée ») sont enregistrés ensemble à la conversion et sont
    // synonymes : on masque le second pour éviter la répétition.
    const hasConvertedEvent = apiEvents.some(
      (ev) => ev.event_type === 'request_converted'
    );
    apiEvents.forEach((ev) => {
      if (ev.event_type === 'booking_created' && hasConvertedEvent) return;
      pushEvent({
        event: ev.label || ev.event_type,
        date: ev.created_at,
        type: ev.event_type === 'cancelled' ? 'cancel' : undefined,
        eventId: ev.id,
      });
    });

    // Événements de cycle (créée / envoyée / acceptée / convertie) : uniquement
    // en l'absence de timeline API, pour éviter les doublons de libellés.
    if (!hasApiTimeline) {
      const creator = request?.created_by_name;
      const company = request?.accepted_by_company?.name;
      pushEvent({
        event: `Demande créée${creator ? ` par ${creator}` : ''}`,
        date: request?.created_at,
      });
      pushEvent({ event: 'Envoyée aux transporteurs', date: request?.sent_at });
      pushEvent({
        event: `Acceptée${company ? ` par ${company}` : ''}`,
        date: request?.accepted_at,
      });
      pushEvent({ event: 'Convertie en booking', date: request?.converted_at });
    }

    // Événements opérationnels (prise en charge / dépose par trajet) : toujours
    // ajoutés car absents de la timeline API.
    const journey = Array.isArray(bs?.route_journey) ? bs.route_journey : null;
    if (journey?.length) {
      journey.forEach((ev) => {
        pushEvent({
          event: ev.event,
          date: ev.date,
          type: ev.type,
          eventId: ev.id,
        });
      });
    } else if (!hasApiTimeline) {
      pushEvent({ event: 'Patient pris en charge', date: bs?.boarded_at });
      pushEvent({ event: 'Transport terminé', date: bs?.completed_at });
    }

    const bsCancelled = !hasApiTimeline ? bs?.cancelled_at : null;
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
      pushEvent({ event: detail, date: bsCancelled, type: 'cancel' });
    } else if (!hasApiTimeline && request?.cancelled_at) {
      pushEvent({ event: 'Annulée', date: request.cancelled_at, type: 'cancel' });
    }

    if (!hasApiTimeline && isExternalRequest(request)) {
      const ext = request.external_carrier || {};
      if (ext.assigned_at) {
        pushEvent({
          event: `Transporteur externe affecté${ext.name ? ` — ${ext.name}` : ''}`,
          date: ext.assigned_at,
        });
      }
      if (ext.executed_at) {
        pushEvent({
          event: 'Déclarée réalisée par l\'institution',
          date: ext.executed_at,
        });
      }
    }

    return events
      .filter((it) => it.date)
      .sort((a, b) => new Date(b.date) - new Date(a.date));
  }, [timelineData, request]);

  const institutionRole = meData?.institution_role;
  const canManage = canManageRequests(institutionRole);
  const canBillingEdit = canEditBilling(institutionRole);
  const canViewAmounts = canViewFinancialAmounts(institutionRole);
  const showBillingSection = canViewBillingSection(institutionRole);
  const canExport = canExportTransports(institutionRole);
  const [exportingPdf, setExportingPdf] = useState(null);
  const [isEditingBooking, setIsEditingBooking] = useState(false);
  const [isEditingRequest, setIsEditingRequest] = useState(false);
  const institutionSocket = useMemo(() => getInstitutionSocket(), []);

  const [showSendModal, setShowSendModal] = useState(false);
  const [showAssignExternalForm, setShowAssignExternalForm] = useState(false);
  const [showCompleteExternalForm, setShowCompleteExternalForm] = useState(false);
  const [externalCarrierForm, setExternalCarrierForm] = useState(EMPTY_EXTERNAL_CARRIER_FORM);
  const [externalCompleteNotes, setExternalCompleteNotes] = useState('');
  const [demoChatMessages, setDemoChatMessages] = useState([]);
  const demoTimersRef = useRef([]);
  const isDemoInstitution = useMemo(() => {
    const env = getAuthEnv();
    const mission = (
      localStorage.getItem('demo_recommended_journey') ||
      localStorage.getItem('demo_demo_recommended_journey') ||
      ''
    )
      .toString()
      .trim()
      .toLowerCase();
    return env === 'demo' && mission === 'institution';
  }, []);

  const patchRequestInCache = useCallback((requestTargetId, updater) => {
    queryClient.setQueryData(
      institutionQueryKeys.requestDetail(requestTargetId),
      (oldData) => (oldData ? updater(oldData) : oldData)
    );
    queryClient.setQueriesData({ queryKey: institutionQueryKeys.requests() }, (oldData) => {
      if (!oldData) return oldData;
      if (Array.isArray(oldData)) {
        return oldData.map((row) => (row?.id === requestTargetId ? updater(row) : row));
      }
      if (oldData?.id === requestTargetId) {
        return updater(oldData);
      }
      if (Array.isArray(oldData?.requests)) {
        return {
          ...oldData,
          requests: oldData.requests.map((row) => (row?.id === requestTargetId ? updater(row) : row)),
        };
      }
      if (Array.isArray(oldData?.items)) {
        return {
          ...oldData,
          items: oldData.items.map((row) => (row?.id === requestTargetId ? updater(row) : row)),
        };
      }
      return oldData;
    });
  }, [queryClient]);

  const readDemoSessionState = useCallback(() => {
    try {
      const raw = window.sessionStorage.getItem(DEMO_INSTITUTION_SESSION_KEY);
      return raw ? JSON.parse(raw) : null;
    } catch {
      return null;
    }
  }, []);

  const writeDemoSessionState = useCallback((nextState) => {
    try {
      window.sessionStorage.setItem(DEMO_INSTITUTION_SESSION_KEY, JSON.stringify(nextState));
    } catch {
      // ignore
    }
  }, []);

  const scheduleDemoLifecycle = useCallback((requestSnapshot) => {
    if (!requestSnapshot?.id) return;
    demoTimersRef.current.forEach((timerId) => window.clearTimeout(timerId));
    demoTimersRef.current = [];

    const requestTargetId = requestSnapshot.id;
    const nowIso = new Date().toISOString();
    let demoState = {
      requestId: requestTargetId,
      status: requestSnapshot.status,
      sent_at: requestSnapshot.sent_at,
      accepted_at: requestSnapshot.accepted_at,
      converted_at: requestSnapshot.converted_at,
      booking_id: requestSnapshot.booking_id || Number(`${requestTargetId}01`),
      booking_summary: requestSnapshot.booking_summary || {},
      accepted_by_company: requestSnapshot.accepted_by_company,
      demoChatMessages: [],
    };
    const persisted = readDemoSessionState();
    if (persisted?.requestId === requestTargetId) {
      demoState = { ...demoState, ...persisted };
    }

    const applyDemoState = (updates, notificationMessage = null) => {
      demoState = {
        ...demoState,
        ...updates,
        booking_summary: {
          ...(demoState.booking_summary || {}),
          ...(updates.booking_summary || {}),
        },
      };
      patchRequestInCache(requestTargetId, (prev) => ({
        ...prev,
        ...updates,
        booking_summary: {
          ...(prev.booking_summary || {}),
          ...(updates.booking_summary || {}),
        },
      }));
      if (Array.isArray(demoState.demoChatMessages)) {
        setDemoChatMessages(demoState.demoChatMessages);
      }
      writeDemoSessionState(demoState);
      if (notificationMessage) toast.success(notificationMessage);
    };

    applyDemoState({
      status: 'SENT',
      sent_at: demoState.sent_at || nowIso,
    });

    const pushTimer = (delayMs, callback) => {
      const timerId = window.setTimeout(callback, delayMs);
      demoTimersRef.current.push(timerId);
    };

    pushTimer(20000, () => {
      const acceptedAt = new Date().toISOString();
      applyDemoState(
        {
          status: 'ACCEPTED',
          accepted_at: demoState.accepted_at || acceptedAt,
          accepted_by_company: demoState.accepted_by_company || { name: DEMO_COMPANY_NAME },
          booking_id: demoState.booking_id || Number(`${requestTargetId}01`),
          booking_summary: {
            status: 'PENDING',
            scheduled_time: requestSnapshot.scheduled_time,
            customer_name:
              demoState.booking_summary?.customer_name
              || `${requestSnapshot.patient?.first_name || ''} ${requestSnapshot.patient?.last_name || ''}`.trim(),
          },
        },
        `${DEMO_COMPANY_NAME} a accepté la demande. Discussion ouverte.`
      );
    });

    pushTimer(45000, () => {
      const convertedAt = new Date().toISOString();
      applyDemoState(
        {
          status: 'CONVERTED',
          converted_at: demoState.converted_at || convertedAt,
          booking_id: demoState.booking_id || Number(`${requestTargetId}01`),
          booking_summary: {
            status: 'PENDING',
            scheduled_time: requestSnapshot.scheduled_time,
            customer_name:
              demoState.booking_summary?.customer_name
              || `${requestSnapshot.patient?.first_name || ''} ${requestSnapshot.patient?.last_name || ''}`.trim(),
          },
        },
        'Course créée.'
      );
    });

    pushTimer(60000, () => {
      applyDemoState(
        {
          status: 'CONVERTED',
          booking_summary: {
            status: 'ASSIGNED',
            assigned_at: new Date().toISOString(),
          },
        },
        'Chauffeur assigné.'
      );
    });

    pushTimer(90000, () => {
      const companyMessage = {
        id: `demo-company-${Date.now()}`,
        sender_type: 'COMPANY',
        sender_label: DEMO_COMPANY_NAME,
        content: 'Le chauffeur arrive dans 15 min.',
        created_at: new Date().toISOString(),
      };
      applyDemoState(
        {
          status: 'CONVERTED',
          booking_summary: {
            status: 'EN_ROUTE',
            en_route_at: companyMessage.created_at,
          },
          demoChatMessages: [...(demoState.demoChatMessages || []), companyMessage],
        },
        'Transport en route : le chauffeur arrive dans 15 min.'
      );
      try {
        // La démo institution est terminée : on évite tout redémarrage du guide.
        window.sessionStorage.setItem(DEMO_INSTITUTION_COMPLETED_KEY, '1');
        window.localStorage.removeItem('demo_recommended_journey');
        window.localStorage.removeItem('demo_demo_recommended_journey');
      } catch {
        // ignore
      }
    });
  }, [patchRequestInCache, readDemoSessionState, writeDemoSessionState]);

  const handleSend = useCallback(async () => {
    try {
      await sendMutation.mutateAsync({ requestId: request.id, options: {} });
      setShowSendModal(false);
      toast.success('Demande envoyée');
      if (isDemoInstitution) {
        scheduleDemoLifecycle(request);
      }
    } catch (err) {
      toast.error(err?.response?.data?.error || 'Erreur lors de l\'envoi');
    }
  }, [sendMutation, request, isDemoInstitution, scheduleDemoLifecycle]);

  useEffect(() => {
    if (!isDemoInstitution || !request?.id) return;
    const persisted = readDemoSessionState();
    if (!persisted || persisted.requestId !== request.id) return;
    if (Array.isArray(persisted.demoChatMessages)) {
      setDemoChatMessages(persisted.demoChatMessages);
    }
    patchRequestInCache(request.id, (prev) => ({
      ...prev,
      ...persisted,
      booking_summary: {
        ...(prev.booking_summary || {}),
        ...(persisted.booking_summary || {}),
      },
    }));
  }, [
    isDemoInstitution,
    request?.id,
    request?.status,
    request?.booking_summary?.status,
    readDemoSessionState,
    patchRequestInCache,
  ]);

  useEffect(() => {
    return () => {
      demoTimersRef.current.forEach((timerId) => window.clearTimeout(timerId));
      demoTimersRef.current = [];
    };
  }, []);

  const handleCancel = async () => {
    try {
      await cancelMutation.mutateAsync({ requestId: request.id, reason: '' });
      toast.success('Demande annulée');
    } catch (err) {
      const data = err?.response?.data;
      if (err?.response?.status === 409 && data?.resulting_booking_id) {
        const reason = window.prompt(
          'Motif d\'annulation (obligatoire, min. 10 caractères si en route) :',
          ''
        );
        if (!reason) return;
        try {
          await cancelBookingMutation.mutateAsync({
            bookingId: data.resulting_booking_id,
            data: {
              version: request.booking_summary?.edit_version || 1,
              reason,
              reason_code: 'CLIENT_REQUEST',
            },
          });
          toast.success('Transport annulé');
        } catch (e2) {
          toast.error(e2?.response?.data?.error || 'Erreur annulation transport');
        }
        return;
      }
      toast.error(data?.error || 'Erreur lors de l\'annulation');
    }
  };

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

  const handleExportMissionPdf = useCallback(async (variant) => {
    if (!request?.id) return;
    setExportingPdf(variant);
    try {
      await exportRequestMissionPdf(request.id, { variant });
      toast.success(variant === 'operational' ? 'Bon de transport généré' : 'Rapport de mission généré');
    } catch (err) {
      toast.error(err?.message || 'Erreur lors de l\'export PDF');
    } finally {
      setExportingPdf(null);
    }
  }, [request?.id]);

  const handleComposeCarrierEmail = useCallback(async () => {
    const email = request?.external_carrier?.email;
    if (!email || !request?.id) return;
    setExportingPdf('operational');
    try {
      await exportRequestMissionPdf(request.id, { variant: 'operational' });
      toast.success('Bon téléchargé — joignez-le à l\'e-mail');
    } catch (err) {
      toast.error(err?.message || 'Erreur lors de l\'export du bon');
    } finally {
      setExportingPdf(null);
    }
    const institutionName = meData?.institution?.name || meData?.name || '';
    window.location.href = buildCarrierMailto(email, request, {
      institutionName,
      institutionPhone: meData?.contact_phone,
    });
  }, [request, meData]);

  const demoFetchMessages = useCallback(async (bookingId, options = {}) => {
    let base = { messages: [], has_more: false };
    try {
      base = await fetchBookingMessages(bookingId, options);
    } catch {
      // En mode démo, un booking simulé peut ne pas exister côté API.
    }
    const merged = [...(base.messages || []), ...demoChatMessages];
    const dedup = Array.from(new Map(merged.map((m) => [m.id, m])).values());
    dedup.sort((a, b) => new Date(a.created_at) - new Date(b.created_at));
    return { messages: dedup, has_more: false };
  }, [demoChatMessages]);
  const demoSendMessage = useCallback(async (bookingId, content) => {
    try {
      return await sendBookingMessage(bookingId, content);
    } catch {
      const fallbackMsg = {
        id: `demo-institution-${Date.now()}`,
        sender_type: 'INSTITUTION',
        sender_label: 'Institution',
        content,
        created_at: new Date().toISOString(),
      };
      setDemoChatMessages((prev) => [...prev, fallbackMsg]);
      return { message: fallbackMsg };
    }
  }, []);
  const shouldUseDemoChat = Boolean(isDemoInstitution && (demoChatMessages.length > 0 || request?.booking_id || request?.booking_summary?.id));

  // Loading / Error
  if (isLoading) return (
    <div className={s.panel}>
      <div className={s.panelHeader}>
        <span className={s.panelTitle}>Chargement...</span>
        <button className={s.closeBtn} onClick={onClose} aria-label="Fermer"><HiOutlineX /></button>
      </div>
      <div className={s.panelLoading}>Chargement du détail...</div>
    </div>
  );

  if (error || !request) return (
    <div className={s.panel}>
      <div className={s.panelHeader}>
        <span className={s.panelTitle}>Erreur</span>
        <button className={s.closeBtn} onClick={onClose} aria-label="Fermer"><HiOutlineX /></button>
      </div>
      <div className={s.panelLoading}>Demande non trouvée</div>
    </div>
  );

  const isExternal = isExternalRequest(request);
  const bs = hasBooking(request) ? request.booking_summary : null;
  const bookingIdForOperations = bs?.id || null;
  const isConverted = hasBooking(request);
  const chatBookingId =
    bookingIdForOperations || (isDemoInstitution && request.id ? Number(`${request.id}01`) : null);
  const canShowChat =
    Boolean(chatBookingId)
    && !isExternal
    && (isConverted || (isDemoInstitution && ['ACCEPTED', 'CONVERTED'].includes(request.status)));
  const bookingStatusKey = bs ? resolveBookingStatusKey(bs) : '';
  const isBoarded = Boolean(bs?.boarded_at);
  const canEditBookingOperational = Boolean(
    canManage
    && isConverted
    && !isExternal
    && !isBoarded
    && !['COMPLETED', 'RETURN_COMPLETED', 'CANCELED'].includes(bookingStatusKey)
  );
  const isBookingEnRoute = bookingStatusKey === 'EN_ROUTE';
  const canEditRequestNow = Boolean(
    canManage
    && !isConverted
    && (
      ['DRAFT', 'SENT', 'ACCEPTED'].includes(request.status)
      || request.status === EXTERNAL_STATUSES.ASSIGNED
    )
  );
  const showAssignExternalAction = canManage && canAssignExternalCarrier(request);
  const showCompleteExternalAction = canManage && canCompleteExternalMission(request);
  const patientName = request.patient
    ? `${request.patient.first_name} ${request.patient.last_name}`
    : bs?.customer_name || '—';
  const routePoints = getRoutePoints(request);
  const tripBadge = getTripBadge(request, routePoints);

  return (
    <div className={s.panel} data-tour-id="institution-request-detail-panel">
      {/* ── Panel header ── */}
      <div className={s.panelHeader}>
        <div className={s.panelTitleRow}>
          <span className={s.panelTitle}>Demande #{request.id}</span>
        </div>
        {canExport && (
          <div className={s.pdfExportGroup}>
            <button
              type="button"
              className={s.pdfExportBtn}
              disabled={Boolean(exportingPdf)}
              onClick={() => handleExportMissionPdf('operational')}
              title="Bon de transport (1 page)"
            >
              <FaFilePdf size={11} />
              {exportingPdf === 'operational' ? '…' : 'Bon'}
            </button>
            <button
              type="button"
              className={s.pdfExportBtn}
              disabled={Boolean(exportingPdf)}
              onClick={() => handleExportMissionPdf('audit')}
              title="Rapport de mission (audit)"
            >
              <FaFilePdf size={11} />
              {exportingPdf === 'audit' ? '…' : 'Rapport'}
            </button>
          </div>
        )}
        <button className={s.closeBtn} onClick={onClose} aria-label="Fermer"><HiOutlineX /></button>
      </div>

      {/* ── Scrollable content ── */}
      <div className={s.panelBody}>

        {bs?.pending_change_request?.status === 'escalation_required' && canManage && bookingIdForOperations && (
          <div className={s.actions} style={{ marginBottom: 12 }}>
            <p style={{ margin: 0, fontSize: 13, color: '#b45309' }}>
              Escalade requise — la validation transporteur a expiré. Remettez la course en diffusion.
            </p>
            <button
              type="button"
              className={`${s.actionBtn} ${s.btnSecondary}`}
              disabled={redispatchMutation.isPending}
              onClick={async () => {
                try {
                  await redispatchMutation.mutateAsync({ bookingId: bookingIdForOperations });
                  toast.success('Course remise en diffusion');
                  queryClient.invalidateQueries({ queryKey: institutionQueryKeys.requestDetail(requestId) });
                } catch (err) {
                  toast.error(err?.response?.data?.error || 'Échec de la rediffusion');
                }
              }}
            >
              Remettre en diffusion
            </button>
          </div>
        )}

        {/* Actions */}
        {canEditBookingOperational && !isEditingBooking && (
          <div className={s.actions}>
            <button
              className={`${s.actionBtn} ${s.btnSecondary}`}
              onClick={() => setIsEditingBooking(true)}
              title="Modifier le transport"
            >
              <FaEdit size={11} /> Modifier
            </button>
            <button
              className={`${s.actionBtn} ${s.btnDanger}`}
              onClick={handleCancel}
              disabled={cancelBookingMutation.isPending}
              title="Annuler le transport"
            >
              <FaTimes size={11} /> Annuler
            </button>
          </div>
        )}

        {isEditingBooking && canEditBookingOperational && (
          <InstitutionOperationalEdit
            request={request}
            bookingId={bookingIdForOperations}
            editVersion={bs?.edit_version || 1}
            isEnRoute={isBookingEnRoute}
            onCancel={() => setIsEditingBooking(false)}
            onSaved={() => {
              setIsEditingBooking(false);
              toast.success('Transport mis à jour');
            }}
            patchMutation={patchBookingMutation}
          />
        )}

        {canEditRequestNow && !isEditingRequest && (
          <div className={s.actions}>
            {!isExternal && (
              <button
                className={`${s.actionBtn} ${s.btnSecondary}`}
                onClick={() => setIsEditingRequest(true)}
                title="Modifier la demande"
              >
                <FaEdit size={11} /> Modifier
              </button>
            )}
            {request.status === 'DRAFT' && !isExternal && (
              <button
                className={`${s.actionBtn} ${s.btnPrimary}`}
                onClick={() => setShowSendModal(true)}
                disabled={sendMutation.isPending}
                data-tour-id="institution-request-send-btn"
              >
                <FaPaperPlane size={11} /> Envoyer
              </button>
            )}
            {showAssignExternalAction && !showAssignExternalForm && (
              <button
                type="button"
                className={`${s.actionBtn} ${s.btnSecondary}`}
                onClick={() => setShowAssignExternalForm(true)}
                disabled={assignExternalMutation.isPending}
                title="Affecter un transporteur externe"
              >
                <FaTruck size={11} /> Externe
              </button>
            )}
            {showCompleteExternalAction && !showCompleteExternalForm && (
              <button
                type="button"
                className={`${s.actionBtn} ${s.btnPrimary}`}
                onClick={() => setShowCompleteExternalForm(true)}
                disabled={completeExternalMutation.isPending}
              >
                <FaTruck size={11} /> Déclarer réalisée
              </button>
            )}
            <button
              className={`${s.actionBtn} ${s.btnDanger}`}
              onClick={handleCancel}
              disabled={cancelMutation.isPending}
            >
              <FaTimes size={11} /> Annuler
            </button>
          </div>
        )}

        {showAssignExternalAction && showAssignExternalForm && (
          <div className={s.section}>
            <ExternalCarrierFields
              value={externalCarrierForm}
              onChange={setExternalCarrierForm}
              idPrefix={`assign-external-${request.id}`}
            />
            <div className={s.actions}>
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
                {assignExternalMutation.isPending ? '…' : 'Confirmer l\'affectation'}
              </button>
            </div>
          </div>
        )}

        {showCompleteExternalAction && showCompleteExternalForm && (
          <div className={s.section}>
            <p className={s.billingMuted}>
              Déclaration manuelle : la mission sera marquée comme réalisée par l&apos;institution.
            </p>
            <label htmlFor={`complete-external-notes-${request.id}`} className={s.billingMuted}>
              Notes (optionnel)
            </label>
            <textarea
              id={`complete-external-notes-${request.id}`}
              className={s.editInput}
              rows={3}
              value={externalCompleteNotes}
              onChange={(e) => setExternalCompleteNotes(e.target.value)}
              placeholder="Commentaire interne"
            />
            <div className={s.actions}>
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
                {completeExternalMutation.isPending ? '…' : 'Confirmer la déclaration'}
              </button>
            </div>
          </div>
        )}

        {canEditRequestNow && isEditingRequest && !isExternal && (
          <InstitutionRequestEdit
            request={request}
            onCancel={() => setIsEditingRequest(false)}
            onSaved={({ carrierNotified } = {}) => {
              setIsEditingRequest(false);
              toast.success(
                carrierNotified
                  ? 'Modification enregistrée — les transporteurs en attente sont informés.'
                  : 'Demande mise à jour',
              );
            }}
          />
        )}

        {/* Transport LIRIE (booking) */}
        {isConverted && !isExternal && bs && (() => {
          const carrier = request.accepted_by_company || {};
          const carrierName = carrier.name || 'Transport';
          const carrierPhone = (carrier.contact_phone || '').toString().trim();
          const carrierEmail = (carrier.contact_email || '').toString().trim();
          const phoneHref = carrierPhone
            ? `tel:${carrierPhone.replace(/[^+0-9]/g, '')}`
            : '';
          return (
            <div className={s.section}>
              <div className={s.sectionHeader}>
                <div className={`${s.sectionIcon} ${s.sectionIconBrand}`}><FaTruck /></div>
                <h3 className={s.sectionTitle}>{carrierName}</h3>
              </div>
              {(carrierPhone || carrierEmail) && (
                <div className={s.carrierContacts}>
                  {carrierPhone && (
                    <a
                      href={phoneHref}
                      className={s.carrierContactItem}
                      title={`Appeler ${carrierName}`}
                      aria-label={`Appeler ${carrierName} au ${carrierPhone}`}
                    >
                      <FaPhoneAlt aria-hidden="true" />
                      <span>{carrierPhone}</span>
                    </a>
                  )}
                  {carrierEmail && (
                    <a
                      href={`mailto:${carrierEmail}`}
                      className={s.carrierContactItem}
                      title={`Écrire à ${carrierName}`}
                      aria-label={`Envoyer un email à ${carrierName} (${carrierEmail})`}
                    >
                      <FaEnvelope aria-hidden="true" />
                      <span>{carrierEmail}</span>
                    </a>
                  )}
                </div>
              )}
              <div className={s.summaryGrid}>
                <div className={s.summaryItem}>
                  <span className={s.summaryLabel}>Horaire</span>
                  <span className={s.summaryValue}>{fmt(bs.scheduled_time)}</span>
                </div>
                <div className={s.summaryItem}>
                  <span className={s.summaryLabel}>Patient</span>
                  <span className={s.summaryValue}>{patientName}</span>
                </div>
                <div className={s.summaryItem}>
                  <span className={s.summaryLabel}>Statut</span>
                  <span className={s.summaryValue}>
                    {BOOKING_STATUS_LABELS[resolveBookingStatusKey(bs)] || 'En cours'}
                  </span>
                </div>
                {canViewAmounts && bs.amount != null && (
                  <div className={s.summaryItem}>
                    <span className={s.summaryLabel}>Montant</span>
                    <span className={s.summaryValue}>{Number(bs.amount).toFixed(2)} CHF</span>
                  </div>
                )}
              </div>
            </div>
          );
        })()}

        {/* Transporteur externe */}
        {isExternal && (
          <ExternalCarrierSection
            request={request}
            onComposeEmail={handleComposeCarrierEmail}
            composing={exportingPdf === 'operational'}
          />
        )}

        {/* Route (masquée pendant l'édition : l'éditeur de parcours la remplace) */}
        {!isEditingRequest && (
        <div className={s.section}>
          <div className={s.sectionHeader}>
            <div className={`${s.sectionIcon} ${s.sectionIconBrand}`}><FaRoute /></div>
            <h3 className={s.sectionTitle}>Trajet</h3>
          </div>
          <div className={s.route}>
            {routePoints.map((point, index) => {
              const isFirst = index === 0;
              const isLast = index === routePoints.length - 1;
              const dotClass = isFirst ? s.routeDotStart : isLast ? s.routeDotEnd : s.routeDotMid;
              const hasDetails = point.details
                && (point.details.establishment || point.details.service || point.details.doctor);
              return (
                <div className={s.routeStop} key={`stop-${point.kind}-${index}-${point.address || ''}`}>
                  <div className={s.routeMarker}>
                    <span className={`${s.routeDot} ${dotClass}`} />
                    {!isLast && <span className={s.routeConnector} />}
                  </div>
                  <div className={s.routeStopBody}>
                    <div className={s.routeStopLabel}>
                      {point.label}
                      {point.timeLabel ? (
                        <span className={s.routeStopTime}> · {point.timeLabel}</span>
                      ) : null}
                    </div>
                    <div className={s.routeStopAddress}>{point.address || '—'}</div>
                    {hasDetails && (
                      <div className={s.routeStopDetails}>
                        {[point.details.establishment, point.details.service, point.details.doctor]
                          .filter(Boolean)
                          .join(' · ')}
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
          <div className={s[tripBadge.className]}>{tripBadge.label}</div>
        </div>
        )}

        {/* Détails */}
        <div className={s.section}>
          <div className={s.sectionHeader}>
            <div className={`${s.sectionIcon} ${s.sectionIconBlue}`}><FaInfoCircle /></div>
            <h3 className={s.sectionTitle}>Détails</h3>
          </div>
          <div className={s.infoRow}>
            <span className={s.infoLabel}>Type</span>
            <span className={s.infoValue}>{MISSION_LABELS[request.mission_type] || request.mission_type}</span>
          </div>
          <div className={s.infoRow}>
            <span className={s.infoLabel}>Date et heure</span>
            <span className={s.infoValue}>{fmt(request.scheduled_time)}</span>
          </div>
          <div className={s.infoRow}>
            <span className={s.infoLabel}>Type de trajet</span>
            <span className={s.infoValue}>{tripBadge.label}</span>
          </div>
          {(request?.is_round_trip || request?.round_trip) && (
            <div className={s.infoRow}>
              <span className={s.infoLabel}>Retour</span>
              <span className={s.infoValue}>
                {formatReturnTimeLabel(request) || '—'}
              </span>
            </div>
          )}
          {request.external_reference && (
            <div className={s.infoRow}>
              <span className={s.infoLabel}>Réf. externe</span>
              <span className={`${s.infoValue} ${s.infoMono}`}>{request.external_reference}</span>
            </div>
          )}
        </div>

        {/* Besoins spécifiques */}
        {(() => {
          const mob = request.mobility || {};
          const hasWheelchair = request.requires_wheelchair || mob.wheelchair;
          const hasVehicleWheelchair = mob.vehicle_wheelchair;
          const hasAssistance = request.requires_assistance || mob.needs_assistance;
          const assistanceType = (mob.assistance_type || '').trim();
          const hasAny = hasWheelchair || hasVehicleWheelchair || hasAssistance
            || request.requires_stretcher || request.requires_oxygen || request.notes;
          if (!hasAny) return null;
          return (
          <div className={s.section}>
            <div className={s.sectionHeader}>
              <div className={`${s.sectionIcon} ${s.sectionIconMuted}`}><FaNotesMedical /></div>
              <h3 className={s.sectionTitle}>Besoins</h3>
            </div>
            <div className={s.needsRow}>
              {hasWheelchair && (
                <span className={`${s.needsChip} ${s.needsChipActive}`}><FaWheelchair size={10} /> Fauteuil</span>
              )}
              {hasVehicleWheelchair && (
                <span className={`${s.needsChip} ${s.needsChipActive}`}>Prendre chaise</span>
              )}
              {hasAssistance && (
                <span className={`${s.needsChip} ${s.needsChipActive}`}>Assistance</span>
              )}
              {request.requires_stretcher && (
                <span className={`${s.needsChip} ${s.needsChipActive}`}>Brancard</span>
              )}
              {request.requires_oxygen && (
                <span className={`${s.needsChip} ${s.needsChipDanger}`}>O₂</span>
              )}
            </div>
            {hasAssistance && assistanceType && (
              <div className={s.routeStopDetails} style={{ marginTop: 6 }}>
                Type d'assistance : {assistanceType}
              </div>
            )}
            {request.notes && (
              <div className={s.notesBlock}>{request.notes}</div>
            )}
          </div>
          );
        })()}

        {/* Facturation */}
        {showBillingSection && (canBillingEdit || request.billing_intent) && (
          <BillingSection
            key={request.id}
            request={request}
            canBilling={canBillingEdit}
            billingMutation={billingMutation}
            bookingBillingMutation={bookingBillingMutation}
          />
        )}

        {/* Mini-canal de communication (ouvert dès acceptation en mode démo) */}
        {canShowChat && (
          <BookingChat
            bookingId={chatBookingId}
            socket={institutionSocket}
            fetchMessages={shouldUseDemoChat ? demoFetchMessages : fetchBookingMessages}
            sendMessage={shouldUseDemoChat ? demoSendMessage : sendBookingMessage}
            closed={['COMPLETED', 'RETURN_COMPLETED', 'CANCELED'].includes(resolveBookingStatusKey(bs))}
          />
        )}

        {/* Historique */}
        {(timeline.length > 0 || isConverted || isExternal || timelineLoading) && (
          <div className={s.section}>
            <div className={s.sectionHeader}>
              <div className={`${s.sectionIcon} ${s.sectionIconMuted}`}><FaHistory /></div>
              <h3 className={s.sectionTitle}>Historique</h3>
            </div>
            <div className={s.timeline}>
              {timelineLoading && (
                <p className={s.billingMuted}>Chargement historique…</p>
              )}
              {timeline.map((item) => (
                <div
                  key={item.eventId || `${item.date}-${item.event}`}
                  className={`${s.timelineItem} ${item.type === 'cancel' ? s.timelineItemCancel : ''}`}
                >
                  <div className={s.timelineEvent}>{item.event}</div>
                  <div className={s.timelineDate}>{fmtShort(item.date)}</div>
                </div>
              ))}
            </div>
          </div>
        )}
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

export default RequestDetailPanel;
