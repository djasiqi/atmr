// pages/institution/Requests/InstitutionRequests.jsx
/**
 * Liste des demandes de transport — layout master-detail.
 * Colonne gauche : liste des demandes
 * Colonne droite : panel détail de la demande sélectionnée
 */

import React, { useState, useMemo, useCallback, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { FaSearch, FaFilter, FaPhoneAlt } from 'react-icons/fa';
import { FiAlertTriangle } from 'react-icons/fi';
import { toast } from 'sonner';
import {
  useInstitutionRequests,
  useInstitutionMe,
  useUpdateRequestBilling,
  useUpdateBookingBilling,
} from '../../../hooks/useInstitutionData';
import { canEditBilling } from '../../../utils/institutionPermissions';
import DemoInteractiveGuide from '../../../components/demo/DemoInteractiveGuide';
import { getAuthEnv } from '../../../utils/webAuthSession';
import RequestDetailPanel from './RequestDetailPanel';
import s from './InstitutionRequests.module.css';

// ─── Status maps ───────────────────────────────────────────
const BOOKING_STATUS = {
  PENDING:          { label: 'En attente',       badge: 'badgePending',         indicator: 'indicatorPending' },
  ACCEPTED:         { label: 'Accepté',          badge: 'badgeAccepted',        indicator: 'indicatorAccepted' },
  ASSIGNED:         { label: 'Chauffeur assigné', badge: 'badgeAssigned',       indicator: 'indicatorAssigned' },
  EN_ROUTE:         { label: 'En route',         badge: 'badgeEnRoute',         indicator: 'indicatorEnRoute' },
  IN_PROGRESS:      { label: 'En cours',         badge: 'badgeInProgress',      indicator: 'indicatorInProgress' },
  OUTBOUND_COMPLETED: { label: 'Retour en cours', badge: 'badgeInProgress',     indicator: 'indicatorInProgress' },
  COMPLETED:        { label: 'Terminé',          badge: 'badgeCompleted',       indicator: 'indicatorCompleted' },
  RETURN_COMPLETED: { label: 'Aller-retour OK',  badge: 'badgeReturnCompleted', indicator: 'indicatorReturnCompleted' },
  CANCELED:         { label: 'Annulé',           badge: 'badgeCancelled',       indicator: 'indicatorCancelled' },
};

const REQUEST_STATUS = {
  DRAFT:     { label: 'Brouillon', badge: 'badgeDraft',     indicator: 'indicatorDraft' },
  SENT:      { label: 'Envoyée',   badge: 'badgeSent',      indicator: 'indicatorSent' },
  ACCEPTED:  { label: 'Acceptée',  badge: 'badgeAccepted',  indicator: 'indicatorAccepted' },
  CONVERTED: { label: 'Confirmée', badge: 'badgeConverted', indicator: 'indicatorConverted' },
  CANCELLED: { label: 'Annulée',   badge: 'badgeCancelled', indicator: 'indicatorCancelled' },
  EXPIRED:   { label: 'Expirée',   badge: 'badgeExpired',   indicator: 'indicatorExpired' },
};

const STATUS_FILTER_LABELS = {
  '': 'Toutes',
  DRAFT: 'Brouillon',
  SENT: 'Envoyée',
  CONVERTED: 'Confirmée',
  CANCELLED: 'Annulée',
};

const INSTITUTION_RESUME_STEP_KEY = 'demo_institution_resume_step';
const DEMO_INSTITUTION_COMPLETED_KEY = 'demo_institution_journey_completed';

// ─── Helpers ───────────────────────────────────────────────
const shortAddr = (addr) => {
  if (!addr) return '—';
  return addr;
};

const fmtTime = (d) => {
  if (!d) return '';
  return new Date(d).toLocaleTimeString('fr-CH', { hour: '2-digit', minute: '2-digit' });
};

const fmtDateShort = (d) => {
  if (!d) return '';
  return new Date(d).toLocaleDateString('fr-CH', { day: '2-digit', month: '2-digit', year: 'numeric' });
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

  // Source de vérité préférée pour A/R: statut agrégé backend.
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

  // Défensif: si completed_at existe, l'UI doit afficher "Terminé"
  // même si un statut stale "IN_PROGRESS" arrive encore du backend/cache.
  if (
    bookingSummary.completed_at &&
    !hasReturn &&
    normalized !== 'RETURN_COMPLETED' &&
    normalized !== 'CANCELED'
  ) {
    return 'COMPLETED';
  }

  // Défensif: si le patient est déjà pris en charge (boarded_at), l'UI doit
  // afficher "En cours" même si le statut métier n'a pas encore basculé.
  if (
    bookingSummary.boarded_at &&
    normalized !== 'COMPLETED' &&
    normalized !== 'RETURN_COMPLETED' &&
    normalized !== 'CANCELED'
  ) {
    return 'IN_PROGRESS';
  }

  // De même, si l'événement "en route" est connu mais le statut reste figé.
  if (
    bookingSummary.en_route_at &&
    !bookingSummary.boarded_at &&
    (normalized === 'ACCEPTED' || normalized === 'ASSIGNED' || normalized === '')
  ) {
    return 'EN_ROUTE';
  }

  return normalized;
};

const getDateGroupLabel = (dateStr) => {
  if (!dateStr) return 'Autre';
  const d = new Date(dateStr);
  const now = new Date();
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const tomorrow = new Date(today); tomorrow.setDate(today.getDate() + 1);
  const yesterday = new Date(today); yesterday.setDate(today.getDate() - 1);
  const target = new Date(d.getFullYear(), d.getMonth(), d.getDate());

  if (target.getTime() === today.getTime()) return "Aujourd'hui";
  if (target.getTime() === tomorrow.getTime()) return 'Demain';
  if (target.getTime() === yesterday.getTime()) return 'Hier';
  return d.toLocaleDateString('fr-CH', { weekday: 'long', day: 'numeric', month: 'long' });
};

const resolveStatus = (req) => {
  if (req.status === 'CONVERTED' && req.booking_summary?.status) {
    const bookingStatusKey = resolveBookingStatusKey(req.booking_summary);
    return BOOKING_STATUS[bookingStatusKey] || REQUEST_STATUS.CONVERTED;
  }
  return REQUEST_STATUS[req.status] || REQUEST_STATUS.DRAFT;
};

// ─── Délais ─────────────────────────────────────────────
const TERMINAL_BOOKING_STATUSES = new Set([
  'COMPLETED', 'RETURN_COMPLETED', 'CANCELED', 'CANCELLED',
]);
const TERMINAL_REQUEST_STATUSES = new Set([
  'CANCELLED', 'EXPIRED',
]);

const LATE_THRESHOLD_MIN = 5;
const LATE_SEVERE_MIN = 15;

// Statuts "course non démarrée": seuls cas où l'institution doit être alertée
// d'un retard. Dès que la course est EN_ROUTE / IN_PROGRESS / boarded /
// completed, le suivi est du ressort du transporteur.
const PRE_START_BOOKING_STATUSES = new Set(['PENDING', 'ACCEPTED', 'ASSIGNED']);
const PRE_START_REQUEST_STATUSES = new Set(['DRAFT', 'SENT', 'ACCEPTED']);

const resolveDelayInfo = (req, nowMs) => {
  if (!req || !req.scheduled_time) return null;
  const bs = req.booking_summary;
  const bookingStatus = String(bs?.status || '').toUpperCase();
  const requestStatus = String(req.status || '').toUpperCase();

  if (bs && TERMINAL_BOOKING_STATUSES.has(bookingStatus)) return null;
  if (!bs && TERMINAL_REQUEST_STATUSES.has(requestStatus)) return null;

  // La course est démarrée (patient pris en charge ou course terminée):
  // plus de retard à signaler côté institution.
  if (bs?.boarded_at) return null;
  if (bs?.completed_at) return null;

  // Ne signaler un retard que si la course n'a pas encore démarré.
  if (bs) {
    if (!PRE_START_BOOKING_STATUSES.has(bookingStatus)) return null;
  } else if (!PRE_START_REQUEST_STATUSES.has(requestStatus)) {
    return null;
  }

  const scheduled = new Date(req.scheduled_time).getTime();
  if (Number.isNaN(scheduled)) return null;
  const diffMin = Math.floor((nowMs - scheduled) / 60000);
  if (diffMin < LATE_THRESHOLD_MIN) return null;

  const severity = diffMin >= LATE_SEVERE_MIN ? 'severe' : 'warning';
  return {
    minutesLate: diffMin,
    severity,
    notStarted: true,
    label: `Retard +${diffMin} min`,
    title: `La course n'a pas démarré (prévue à ${new Date(req.scheduled_time).toLocaleTimeString('fr-CH', { hour: '2-digit', minute: '2-digit' })}). Contactez le transporteur.`,
  };
};

const formatPhoneHref = (phone) => {
  if (!phone) return '';
  const cleaned = String(phone).replace(/[^+0-9]/g, '');
  return cleaned ? `tel:${cleaned}` : '';
};

const formatPhoneDisplay = (phone) => {
  if (!phone) return '';
  return String(phone).trim();
};

// ─── Component ─────────────────────────────────────────────
const InstitutionRequests = () => {
  const location = useLocation();
  const [filters, setFilters] = useState({
    status: '', date_from: '', date_to: '', query: '', page: 1, per_page: 20,
  });
  const [showFilters, setShowFilters] = useState(false);
  const [selectedId, setSelectedId] = useState(null);

  const { data: meData } = useInstitutionMe();
  const myRole = meData?.institution_role;
  const showBillingSwitch = canEditBilling(myRole);

  const updateRequestBilling = useUpdateRequestBilling();
  const updateBookingBilling = useUpdateBookingBilling();

  const { data: requestsData, isLoading, error } = useInstitutionRequests(filters);

  const [nowMs, setNowMs] = useState(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNowMs(Date.now()), 60000);
    return () => clearInterval(id);
  }, []);

  const requests = useMemo(
    () => requestsData?.requests || requestsData?.items || [],
    [requestsData]
  );
  const totalPages = Math.ceil((requestsData?.total || 0) / filters.per_page);

  const handleFilter = (key, value) => setFilters(prev => ({ ...prev, [key]: value, page: 1 }));

  const handleSelectRequest = useCallback((req) => {
    setSelectedId((prev) => (prev === req.id ? null : req.id));
  }, []);

  const handleClosePanel = useCallback(() => {
    setSelectedId(null);
  }, []);

  // Group by date
  const grouped = useMemo(() => {
    const groups = {};
    for (const req of requests) {
      const label = getDateGroupLabel(req.scheduled_time);
      if (!groups[label]) groups[label] = [];
      groups[label].push(req);
    }
    return groups;
  }, [requests]);

  // Status counts for pills
  const statusCounts = useMemo(() => {
    const counts = { '': requests.length };
    for (const req of requests) {
      const key = req.status;
      counts[key] = (counts[key] || 0) + 1;
    }
    return counts;
  }, [requests]);

  const handleToggleBilling = useCallback((e, req) => {
    e.stopPropagation(); // ne pas ouvrir le panel
    const isConverted = req.status === 'CONVERTED' && req.booking_summary;
    const currentIntent = isConverted
      ? (req.booking_summary.billed_to_type === 'clinic' ? 'institution' : 'patient')
      : (req.billing_intent || 'patient');
    const newIntent = currentIntent === 'patient' ? 'institution' : 'patient';
    const billingPayload = {
      billing_intent: newIntent,
      override_reason: 'Modification rapide depuis la liste institution',
      billing_change_reason_code: 'ADMIN_CORRECTION',
    };

    if (isConverted && req.booking_summary.id) {
      updateBookingBilling.mutate(
        {
          bookingId: req.booking_summary.id,
          data: {
            ...billingPayload,
            version: req.booking_summary.edit_version,
          },
        },
        {
          onSuccess: () => toast.success(`Facturation → ${newIntent === 'patient' ? 'Patient' : 'Clinique'}`),
          onError: (err) => toast.error(err?.response?.data?.error || 'Erreur facturation'),
        },
      );
    } else {
      updateRequestBilling.mutate(
        { requestId: req.id, data: billingPayload },
        {
          onSuccess: () => toast.success(`Facturation → ${newIntent === 'patient' ? 'Patient' : 'Clinique'}`),
          onError: (err) => toast.error(err?.response?.data?.error || 'Erreur facturation'),
        },
      );
    }
  }, [updateRequestBilling, updateBookingBilling]);

  const panelOpen = selectedId !== null;
  const fallbackDemoMission = useMemo(() => {
    const isDemoEnv = getAuthEnv() === 'demo';
    if (!isDemoEnv) return null;
    return (
      localStorage.getItem('demo_recommended_journey') ||
      localStorage.getItem('demo_demo_recommended_journey') ||
      ''
    )
      .toString()
      .trim()
      .toLowerCase();
  }, []);
  const demoMission = useMemo(() => {
    try {
      if (window.sessionStorage.getItem(DEMO_INSTITUTION_COMPLETED_KEY) === '1') return null;
    } catch {
      // ignore
    }
    const mission = new URLSearchParams(location.search).get('demo_mission');
    if (mission) return mission.toString().trim().toLowerCase();
    if (fallbackDemoMission === 'institution') return 'institution';
    return null;
  }, [location.search, fallbackDemoMission]);
  const initialGuideStepId = useMemo(() => {
    try {
      return window.sessionStorage.getItem(INSTITUTION_RESUME_STEP_KEY);
    } catch {
      return null;
    }
  }, []);

  return (
    <div className={`${s.masterDetail} ${panelOpen ? s.masterDetailOpen : ''}`}>
      {demoMission === 'institution' && initialGuideStepId && (
        <DemoInteractiveGuide
          role="institution"
          initialStepId={initialGuideStepId}
          onFinish={() => {
            try {
              window.sessionStorage.removeItem(INSTITUTION_RESUME_STEP_KEY);
            } catch {
              // ignore
            }
          }}
        />
      )}

      {/* ═══ LEFT: Request list ═══ */}
      <div className={s.listColumn} data-tour-id="institution-history">
        {/* Toolbar */}
        <div className={s.toolbar}>
          <div className={s.searchBox}>
            <FaSearch className={s.searchIcon} />
            <input
              type="text"
              placeholder="Rechercher un patient, une référence..."
              value={filters.query}
              onChange={(e) => handleFilter('query', e.target.value)}
            />
          </div>
          <button
            className={`${s.filterBtn} ${showFilters ? s.filterBtnActive : ''}`}
            onClick={() => setShowFilters(!showFilters)}
          >
            <FaFilter size={11} /> Filtres
          </button>
        </div>

        {/* Filters panel */}
        {showFilters && (
          <div className={s.filtersPanel}>
            <div className={s.filterGroup}>
              <label>Statut</label>
              <select value={filters.status} onChange={(e) => handleFilter('status', e.target.value)}>
                <option value="">Tous</option>
                {Object.entries(STATUS_FILTER_LABELS).filter(([k]) => k).map(([val, label]) => (
                  <option key={val} value={val}>{label}</option>
                ))}
              </select>
            </div>
            <div className={s.filterGroup}>
              <label>Du</label>
              <input type="date" value={filters.date_from} onChange={(e) => handleFilter('date_from', e.target.value)} />
            </div>
            <div className={s.filterGroup}>
              <label>Au</label>
              <input type="date" value={filters.date_to} onChange={(e) => handleFilter('date_to', e.target.value)} />
            </div>
            <button
              className={s.clearBtn}
              onClick={() => setFilters({ status: '', date_from: '', date_to: '', query: '', page: 1, per_page: 20 })}
            >
              Effacer
            </button>
          </div>
        )}

        {/* Status pills */}
        <div className={s.statusPills}>
          {Object.entries(STATUS_FILTER_LABELS).map(([val, label]) => (
            <button
              key={val}
              className={`${s.statusPill} ${filters.status === val ? s.statusPillActive : ''}`}
              onClick={() => handleFilter('status', val)}
            >
              {label}
              {statusCounts[val || ''] > 0 && (
                <span className={s.pillCount}>{statusCounts[val || ''] || 0}</span>
              )}
            </button>
          ))}
        </div>

        {/* Request list */}
        {isLoading ? (
          <div className={s.loading}>Chargement...</div>
        ) : error ? (
          <div className={s.error}>Erreur : {error.message}</div>
        ) : requests.length === 0 ? (
          <div className={s.empty}>
            <p>Aucune demande trouvée</p>
            <p className={s.emptyHint}>Modifiez vos filtres ou créez une nouvelle demande.</p>
          </div>
        ) : (
          <>
            {Object.entries(grouped).map(([dateLabel, items]) => (
              <div key={dateLabel} className={s.dateGroup}>
                <div className={s.dateGroupLabel}>{dateLabel}</div>

                {items.map((req) => {
                  const st = resolveStatus(req);
                  const patientName = req.patient
                    ? `${req.patient.last_name} ${req.patient.first_name}`
                    : req.external_reference || `#${req.id}`;
                  const isSelected = selectedId === req.id;

                  const companyName = req.accepted_by_company?.name;
                  const companyPhone = req.accepted_by_company?.contact_phone;
                  const delay = resolveDelayInfo(req, nowMs);
                  const showCallAction = delay && companyPhone;

                  return (
                    <div
                      key={req.id}
                      className={`${s.requestCard} ${isSelected ? s.requestCardSelected : ''} ${delay ? (delay.severity === 'severe' ? s.requestCardLateSevere : s.requestCardLate) : ''}`}
                      data-tour-id={req.status === 'DRAFT' ? 'institution-request-draft-card' : undefined}
                      onClick={() => handleSelectRequest(req)}
                    >
                      <div className={`${s.cardIndicator} ${s[st.indicator]}`} />

                      {/* Col 1 : nom + statut (2 lignes) */}
                      <div className={s.colLeft}>
                        <span className={s.patientName}>{patientName}</span>
                        <div className={s.statusLine}>
                          <span className={`${s.badge} ${s[st.badge]}`}>{st.label}</span>
                          {companyName && <span className={s.companyTag}>{companyName}</span>}
                          {!companyName && req.status === 'SENT' && <span className={s.waitingTag}>En attente</span>}
                          {delay && (
                            <span
                              className={`${s.lateBadge} ${delay.severity === 'severe' ? s.lateBadgeSevere : ''}`}
                              title={delay.title}
                            >
                              <FiAlertTriangle aria-hidden="true" />
                              {delay.label}
                            </span>
                          )}
                        </div>
                      </div>

                      {/* Col 2 : trajet (centré verticalement) */}
                      <div className={s.colCenter}>
                        <span className={`${s.routeDot} ${s.routeDotStart}`} />
                        <span className={s.routeText}>{shortAddr(req.pickup_location)}</span>
                        <span className={s.routeArrow}>→</span>
                        <span className={`${s.routeDot} ${s.routeDotEnd}`} />
                        <span className={s.routeText}>{shortAddr(req.dropoff_location)}</span>
                      </div>

                      {/* Col 3 : date/heure + toggle facturation */}
                      <div className={s.colRight}>
                        <span className={s.cardDateTime}>
                          {fmtDateShort(req.scheduled_time)} · {fmtTime(req.scheduled_time)}
                        </span>
                        {req.scheduled_time_type === 'arrival' && (
                          <span className={s.timeTypeBadge} title="Heure du rendez-vous (arrivée)">📍 RDV</span>
                        )}
                        {showCallAction && (
                          <a
                            href={formatPhoneHref(companyPhone)}
                            className={`${s.callBtn} ${delay.severity === 'severe' ? s.callBtnSevere : ''}`}
                            onClick={(e) => e.stopPropagation()}
                            title={`Appeler ${companyName || 'le transporteur'} — ${formatPhoneDisplay(companyPhone)}`}
                            aria-label={`Appeler ${companyName || 'le transporteur'} au ${formatPhoneDisplay(companyPhone)}`}
                          >
                            <FaPhoneAlt aria-hidden="true" />
                            <span className={s.callBtnLabel}>{formatPhoneDisplay(companyPhone)}</span>
                          </a>
                        )}
                        {showBillingSwitch && (() => {
                          const isConverted = req.status === 'CONVERTED' && req.booking_summary;
                          const isPatient = isConverted
                            ? req.booking_summary.billed_to_type !== 'clinic'
                            : (req.billing_intent || 'patient') === 'patient';
                          const isInvoiced = isConverted && req.booking_summary.is_invoiced;
                          return (
                            <div
                              className={`${s.billingSwitch} ${isPatient ? s.billPatient : s.billClinic} ${isInvoiced ? s.billLocked : ''}`}
                              onClick={isInvoiced ? undefined : (e) => handleToggleBilling(e, req)}
                              title={isInvoiced ? 'Déjà facturé — non modifiable' : 'Cliquez pour changer la facturation'}
                            >
                              <span className={s.billLabel}>{isPatient ? 'Patient' : 'Clinique'}</span>
                              <span className={s.billToggle}>
                                <span className={s.billDot} />
                              </span>
                            </div>
                          );
                        })()}
                      </div>

                    </div>
                  );
                })}
              </div>
            ))}

            {totalPages > 1 && (
              <div className={s.pagination}>
                <button onClick={() => handleFilter('page', filters.page - 1)} disabled={filters.page <= 1}>
                  Précédent
                </button>
                <span>Page {filters.page} / {totalPages}</span>
                <button onClick={() => handleFilter('page', filters.page + 1)} disabled={filters.page >= totalPages}>
                  Suivant
                </button>
              </div>
            )}
          </>
        )}
      </div>

      {/* ═══ RIGHT: Detail panel ═══ */}
      {panelOpen && (
        <div className={s.detailColumn} data-tour-id="institution-request-detail-panel">
          <RequestDetailPanel
            requestId={selectedId}
            onClose={handleClosePanel}
          />
        </div>
      )}
    </div>
  );
};

export default InstitutionRequests;
