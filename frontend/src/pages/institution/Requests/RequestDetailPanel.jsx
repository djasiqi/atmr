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
  FaTruck, FaRoute, FaFileInvoiceDollar,
  FaHistory, FaWheelchair, FaInfoCircle, FaNotesMedical,
  FaPhoneAlt, FaEnvelope,
} from 'react-icons/fa';
import { HiOutlineX } from 'react-icons/hi';
import {
  useInstitutionRequest, useInstitutionMe,
  useSendRequest, useCancelRequest,
  useUpdateRequestBilling, useUpdateBookingBilling,
  usePatchInstitutionBooking, useCancelInstitutionBooking,
  institutionQueryKeys,
} from '../../../hooks/useInstitutionData';
import { useQueryClient } from '@tanstack/react-query';
import {
  canManageRequests,
  canEditBilling,
  canViewFinancialAmounts,
  canViewBillingSection,
} from '../../../utils/institutionPermissions';
import InstitutionOperationalEdit from './InstitutionOperationalEdit';
import InstitutionBookingHistory from './InstitutionBookingHistory';
import InstitutionRequestEdit from './InstitutionRequestEdit';
import { getAuthEnv } from '../../../utils/webAuthSession';
import { toast } from 'sonner';
import { getInstitutionSocket } from '../../../services/institutionSocket';
import { fetchBookingMessages, sendBookingMessage } from '../../../services/institutionService';
import BookingChat from '../../company/Reservations/components/BookingChat';
import s from './RequestDetailPanel.module.css';

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
};

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

// ─── Status badge ──────────────────────────────────────────
const StatusBadge = ({ request }) => {
  if (request.status === 'CONVERTED' && request.booking_summary?.status) {
    const info = BOOKING_STATUS_MAP[resolveBookingStatusKey(request.booking_summary)];
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
  const billingMutation = useUpdateRequestBilling();
  const bookingBillingMutation = useUpdateBookingBilling();
  const patchBookingMutation = usePatchInstitutionBooking();
  const cancelBookingMutation = useCancelInstitutionBooking();

  const institutionRole = meData?.institution_role;
  const canManage = canManageRequests(institutionRole);
  const canBillingEdit = canEditBilling(institutionRole);
  const canViewAmounts = canViewFinancialAmounts(institutionRole);
  const showBillingSection = canViewBillingSection(institutionRole);
  const [isEditingBooking, setIsEditingBooking] = useState(false);
  const [isEditingRequest, setIsEditingRequest] = useState(false);
  const institutionSocket = useMemo(() => getInstitutionSocket(), []);

  const [showSendModal, setShowSendModal] = useState(false);
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

  // Timeline
  const getTimeline = () => {
    if (!request) return [];
    const events = [];
    const creator = request.created_by_name;
    const company = request.accepted_by_company?.name;
    const bs = request.booking_summary;

    const institutionName = request.institution_name || request.institution?.name;
    if (request.created_at)
      events.push({
        event: `Demande créée${creator ? ` par ${creator}` : ''}${institutionName ? ` (${institutionName})` : ''}`,
        date: request.created_at,
      });
    if (request.sent_at)
      events.push({ event: 'Demande envoyée', date: request.sent_at });
    if (request.accepted_at)
      events.push({
        event: `Demande acceptée${company ? ` par ${company}` : ''}`,
        date: request.accepted_at,
      });
    if (request.converted_at)
      events.push({ event: 'Réservation créée', date: request.converted_at });
    if (bs?.assigned_at)
      events.push({ event: 'Chauffeur assigné', date: bs.assigned_at });
    if (bs?.en_route_at)
      events.push({ event: 'En route', date: bs.en_route_at });
    if (bs?.boarded_at)
      events.push({ event: 'Patient pris en charge', date: bs.boarded_at });
    if (bs?.completed_at)
      events.push({ event: 'Transport terminé', date: bs.completed_at });
    const bsCancelled = bs?.cancelled_at;
    if (bsCancelled) {
      const roleMap = { company: 'Entreprise', driver: 'Chauffeur', admin: 'Admin', system: 'Système' };
      const byLabel = roleMap[bs.cancelled_by_role] || '';
      const reasonLabel = bs.cancellation_display_label || '';
      const billable = bs.is_cancellation_billable;

      let detail = 'Annulée';
      if (byLabel) detail += ` par ${byLabel}`;
      if (reasonLabel) detail += ` — ${reasonLabel}`;
      if (billable === true) detail += ' (facturée)';
      else if (billable === false) detail += ' (non facturée)';

      events.push({ event: detail, date: bsCancelled, type: 'cancel' });
    } else if (request.cancelled_at) {
      events.push({ event: 'Annulée', date: request.cancelled_at, type: 'cancel' });
    }

    return events.sort((a, b) => new Date(b.date) - new Date(a.date));
  };

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

  const bs = request.booking_summary;
  const bookingIdForOperations = bs?.id || request.booking_id || null;
  const isConverted = request.status === 'CONVERTED' && Boolean(bookingIdForOperations);
  const chatBookingId =
    bookingIdForOperations || (isDemoInstitution && request.id ? Number(`${request.id}01`) : null);
  const canShowChat =
    Boolean(chatBookingId)
    && (isConverted || (isDemoInstitution && ['ACCEPTED', 'CONVERTED'].includes(request.status)));
  const timeline = getTimeline();
  const bookingStatusKey = bs ? resolveBookingStatusKey(bs) : '';
  const isBoarded = Boolean(bs?.boarded_at);
  const canEditBookingOperational = Boolean(
    canManage
    && isConverted
    && !isBoarded
    && !['COMPLETED', 'RETURN_COMPLETED', 'CANCELED'].includes(bookingStatusKey)
  );
  const isBookingEnRoute = bookingStatusKey === 'EN_ROUTE';
  const canEditRequestNow = Boolean(
    canManage
    && !isConverted
    && ['DRAFT', 'SENT', 'ACCEPTED'].includes(request.status)
  );
  const patientName = request.patient
    ? `${request.patient.first_name} ${request.patient.last_name}`
    : bs?.customer_name || '—';

  return (
    <div className={s.panel} data-tour-id="institution-request-detail-panel">
      {/* ── Panel header ── */}
      <div className={s.panelHeader}>
        <div className={s.panelTitleRow}>
          <span className={s.panelTitle}>Demande #{request.id}</span>
          <StatusBadge request={request} />
        </div>
        <button className={s.closeBtn} onClick={onClose} aria-label="Fermer"><HiOutlineX /></button>
      </div>

      {/* ── Scrollable content ── */}
      <div className={s.panelBody}>

        {/* Actions */}
        {canEditBookingOperational && !isEditingBooking && (
          <div className={s.actions}>
            <button
              className={`${s.actionBtn} ${s.btnSecondary}`}
              onClick={() => setIsEditingBooking(true)}
            >
              <FaEdit size={11} /> Modifier le transport
            </button>
            <button
              className={`${s.actionBtn} ${s.btnDanger}`}
              onClick={handleCancel}
              disabled={cancelBookingMutation.isPending}
            >
              <FaTimes size={11} /> Annuler le transport
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
            <button
              className={`${s.actionBtn} ${s.btnSecondary}`}
              onClick={() => setIsEditingRequest(true)}
            >
              <FaEdit size={11} /> Modifier la demande
            </button>
            {request.status === 'DRAFT' && (
              <button
                className={`${s.actionBtn} ${s.btnPrimary}`}
                onClick={() => setShowSendModal(true)}
                disabled={sendMutation.isPending}
                data-tour-id="institution-request-send-btn"
              >
                <FaPaperPlane size={11} /> Envoyer
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

        {canEditRequestNow && isEditingRequest && (
          <InstitutionRequestEdit
            request={request}
            onCancel={() => setIsEditingRequest(false)}
            onSaved={() => {
              setIsEditingRequest(false);
              toast.success('Demande mise à jour');
            }}
          />
        )}

        {/* Transport summary (if booking) */}
        {isConverted && bs && (() => {
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
                    {BOOKING_STATUS_MAP[resolveBookingStatusKey(bs)]?.label || 'En cours'}
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

        {/* Route */}
        <div className={s.section}>
          <div className={s.sectionHeader}>
            <div className={`${s.sectionIcon} ${s.sectionIconBrand}`}><FaRoute /></div>
            <h3 className={s.sectionTitle}>Trajet</h3>
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
          {(request.is_round_trip || request.round_trip) ? (
            <div className={s.roundTripBadge}>
              Aller-retour
              {request.return_time ? ` — retour ${fmt(request.return_time)}` : ''}
            </div>
          ) : (
            <div className={s.oneWayBadge}>Aller simple</div>
          )}
        </div>

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
            <span className={s.infoValue}>
              {(request.is_round_trip || request.round_trip)
                ? `Aller-retour${request.return_time ? ` (retour ${fmt(request.return_time)})` : ''}`
                : 'Aller simple'}
            </span>
          </div>
          {request.external_reference && (
            <div className={s.infoRow}>
              <span className={s.infoLabel}>Réf. externe</span>
              <span className={`${s.infoValue} ${s.infoMono}`}>{request.external_reference}</span>
            </div>
          )}
        </div>

        {/* Besoins spécifiques */}
        {(request.requires_wheelchair || request.requires_stretcher || request.requires_oxygen || request.notes) && (
          <div className={s.section}>
            <div className={s.sectionHeader}>
              <div className={`${s.sectionIcon} ${s.sectionIconMuted}`}><FaNotesMedical /></div>
              <h3 className={s.sectionTitle}>Besoins</h3>
            </div>
            <div className={s.needsRow}>
              {request.requires_wheelchair && (
                <span className={`${s.needsChip} ${s.needsChipActive}`}><FaWheelchair size={10} /> Fauteuil</span>
              )}
              {request.requires_stretcher && (
                <span className={`${s.needsChip} ${s.needsChipActive}`}>Brancard</span>
              )}
              {request.requires_oxygen && (
                <span className={`${s.needsChip} ${s.needsChipDanger}`}>O₂</span>
              )}
            </div>
            {request.notes && (
              <div className={s.notesBlock}>{request.notes}</div>
            )}
          </div>
        )}

        {/* Facturation */}
        {showBillingSection && (canBillingEdit || request.billing_intent) && (
          <BillingSection
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
        {(timeline.length > 0 || isConverted) && (
          <div className={s.section}>
            <div className={s.sectionHeader}>
              <div className={`${s.sectionIcon} ${s.sectionIconMuted}`}><FaHistory /></div>
              <h3 className={s.sectionTitle}>Historique</h3>
            </div>
            {isConverted && bookingIdForOperations ? (
              <InstitutionBookingHistory
                bookingId={bookingIdForOperations}
                lifecycleTimeline={timeline}
              />
            ) : (
              <div className={s.timeline}>
                {timeline.map((item, i) => (
                  <div key={i} className={`${s.timelineItem} ${item.type === 'cancel' ? s.timelineItemCancel : ''}`}>
                    <div className={s.timelineEvent}>{item.event}</div>
                    <div className={s.timelineDate}>{fmtShort(item.date)}</div>
                  </div>
                ))}
              </div>
            )}
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
