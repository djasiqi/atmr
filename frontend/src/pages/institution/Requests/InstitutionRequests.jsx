// pages/institution/Requests/InstitutionRequests.jsx
/**
 * Liste des demandes de transport — layout master-detail.
 * Colonne gauche : liste des demandes
 * Colonne droite : panel détail de la demande sélectionnée
 */

import React, { useState, useMemo, useCallback, useEffect } from 'react';
import { useLocation, useNavigate, useParams } from 'react-router-dom';
import { FaSearch, FaFilter, FaFilePdf, FaRedo } from 'react-icons/fa';
import { FiAlertTriangle } from 'react-icons/fi';
import { toast } from 'sonner';
import {
  useInstitutionRequests,
  useInstitutionMe,
  useUpdateRequestBilling,
  useUpdateBookingBilling,
  useSendRequest,
} from '../../../hooks/useInstitutionData';
import { canEditBilling, canExportTransports, canManageRequests } from '../../../utils/institutionPermissions';
import { exportDailyMissionReportsZip } from '../../../services/institutionService';
import DemoInteractiveGuide from '../../../components/demo/DemoInteractiveGuide';
import InlineDatePicker from '../../../components/ui/InlineDatePicker';
import ChipSelect from '../../../components/ui/ChipSelect';
import { getAuthEnv } from '../../../utils/webAuthSession';
import { resolveBookingStatusKey } from '../../../utils/institutionBookingStatus';
import RequestDetailPanel from './RequestDetailPanel';
import ConfirmSendModal from './ConfirmSendModal';
import MissionScheduleCardTime from '../../../components/institution/MissionScheduleCardTime';
import { canRelaunchInstitutionRequest } from '../../../utils/institutionRequestDispatch';
import { formatReturnTimeLabel, formatLegScheduleSummary, getNextConfirmedLegTime } from '../../../utils/formatLegTime';
import { getCarrierDisplay } from '../../../utils/carrierDisplay';
import {
  isExternalRequest,
  isConvertedLirie,
  getCarrierSourceLabel,
  EXTERNAL_STATUSES,
} from '../../../utils/requestStatus';
import {
  resolveStatusDisplay,
  buildCardMeta,
  resolveBillingMetaLabel,
} from './statusColors';
import {
  extractWallClockDate,
  extractWallClockTime,
  getGenevaTodayDateStr,
  minutesSinceMissionWallClock,
} from '../../../utils/missionTimeDisplay';
import s from './InstitutionRequests.module.css';

// ─── Filtres statut ───────────────────────────────────────────
const STATUS_FILTER_LABELS = {
  '': 'Toutes',
  DRAFT: 'Brouillon',
  SENT: 'Envoyée',
  CONVERTED: 'Confirmée',
  [EXTERNAL_STATUSES.ASSIGNED]: 'Externe affecté',
  [EXTERNAL_STATUSES.COMPLETED]: 'Externe réalisée',
  CANCELLED: 'Annulée',
};

const STATUS_FILTER_OPTIONS = Object.entries(STATUS_FILTER_LABELS).map(([value, label]) => ({
  value,
  label,
}));

const CARRIER_MODE_FILTER_LABELS = {
  '': 'Tous',
  lirie: 'LIRIE',
  external: 'Externe',
};

const CARRIER_MODE_FILTER_OPTIONS = Object.entries(CARRIER_MODE_FILTER_LABELS).map(([value, label]) => ({
  value,
  label,
}));

const INSTITUTION_RESUME_STEP_KEY = 'demo_institution_resume_step';
const DEMO_INSTITUTION_COMPLETED_KEY = 'demo_institution_journey_completed';

// ─── Helpers ───────────────────────────────────────────────
const shortAddr = (addr) => {
  if (!addr) return '—';
  return addr;
};

const getDateGroupLabel = (dateStr) => {
  if (!dateStr) return 'Autre';
  const targetIso = extractWallClockDate(dateStr);
  if (!targetIso) return 'Autre';

  const todayIso = getGenevaTodayDateStr();
  const [ty, tm, td] = todayIso.split('-').map(Number);
  const tomorrow = new Date(Date.UTC(ty, tm - 1, td + 1));
  const yesterday = new Date(Date.UTC(ty, tm - 1, td - 1));
  const pad = (n) => String(n).padStart(2, '0');
  const tomorrowIso = `${tomorrow.getUTCFullYear()}-${pad(tomorrow.getUTCMonth() + 1)}-${pad(tomorrow.getUTCDate())}`;
  const yesterdayIso = `${yesterday.getUTCFullYear()}-${pad(yesterday.getUTCMonth() + 1)}-${pad(yesterday.getUTCDate())}`;

  if (targetIso === todayIso) return "Aujourd'hui";
  if (targetIso === tomorrowIso) return 'Demain';
  if (targetIso === yesterdayIso) return 'Hier';

  const [y, m, d] = targetIso.split('-').map(Number);
  const labelDate = new Date(Date.UTC(y, m - 1, d, 12, 0, 0));
  return labelDate.toLocaleDateString('fr-CH', {
    weekday: 'long',
    day: 'numeric',
    month: 'long',
    timeZone: 'UTC',
  });
};

const resolveStatus = (req) => resolveStatusDisplay(req, resolveBookingStatusKey);

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
  const effectiveTime = req.next_confirmed_time || getNextConfirmedLegTime(req) || req.scheduled_time;
  if (!req || !effectiveTime) return null;
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

  const diffMin = minutesSinceMissionWallClock(effectiveTime, nowMs);
  if (!Number.isFinite(diffMin) || diffMin < LATE_THRESHOLD_MIN) return null;

  const severity = diffMin >= LATE_SEVERE_MIN ? 'severe' : 'warning';
  const displayTime = extractWallClockTime(effectiveTime);
  return {
    minutesLate: diffMin,
    severity,
    notStarted: true,
    label: `Retard +${diffMin} min`,
    title: `La course n'a pas démarré (prévue à ${displayTime || '—'}). Contactez le transporteur.`,
  };
};

const getRoutePoints = (req) => {
  const legs = Array.isArray(req?.legs)
    ? [...req.legs].sort((a, b) => (a.sequence_index ?? 0) - (b.sequence_index ?? 0))
    : [];
  if (legs.length > 0) {
    return [
      { label: 'Départ', address: legs[0].pickup_location, kind: 'start' },
      ...legs.map((leg, index) => {
        const isReturn = Boolean(req?.return_to_institution) && index === legs.length - 1;
        return {
          label: isReturn ? 'Retour' : `Destination ${index + 1}`,
          address: leg.dropoff_location,
          kind: isReturn ? 'return' : 'destination',
        };
      }),
    ];
  }
  return [
    { label: 'Départ', address: req?.pickup_location, kind: 'start' },
    { label: 'Destination 1', address: req?.dropoff_location, kind: 'destination' },
  ];
};

const isRoundTripRequest = (req) => Boolean(req?.is_round_trip ?? req?.round_trip);

const resolveTripTypeMeta = (req) => {
  const routePoints = getRoutePoints(req);
  if (req?.return_to_institution) {
    return {
      label: 'A/R',
      title: `Parcours aller-retour (${Math.max(routePoints.length - 1, 1)} trajet(s))`,
    };
  }
  if (req?.multi_stop || routePoints.length > 2) {
    const destCount = routePoints.length - 1;
    return {
      label: destCount > 2 ? `${destCount} étapes` : 'Multi-destination',
      title: `${destCount} destination(s) planifiée(s)`,
    };
  }
  if (isRoundTripRequest(req)) {
    const returnHint = formatReturnTimeLabel(req);
    return {
      label: 'A/R',
      title: `Aller-retour${returnHint ? ` — ${returnHint}` : ''}`,
    };
  }
  return {
    label: 'Aller simple',
    title: 'Trajet aller simple (pas de retour planifié)',
  };
};

const parseRouteRequestId = (param) => {
  if (!param) return null;
  const num = Number(param);
  return Number.isFinite(num) ? num : param;
};

// ─── Component ─────────────────────────────────────────────
const InstitutionRequests = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { public_id: publicId, requestId: routeRequestId } = useParams();
  const requestsBasePath = useMemo(() => {
    const isDemoPath = String(location.pathname || '').startsWith('/demo/');
    const dashboardRoot = isDemoPath ? '/demo/dashboard' : '/dashboard';
    return `${dashboardRoot}/institution/${publicId}/requests`;
  }, [location.pathname, publicId]);
  const initialFilters = useMemo(() => {
    const base = {
      status: '',
      carrier_source: '',
      date_from: '',
      date_to: '',
      query: '',
      page: 1,
      per_page: 20,
    };
    const params = new URLSearchParams(location.search);
    if (params.get('day') === 'today') {
      const now = new Date();
      const today = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`;
      return { ...base, date_from: today, date_to: today };
    }
    return base;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  const [filters, setFilters] = useState(initialFilters);
  const [showFilters, setShowFilters] = useState(
    () => initialFilters.date_from !== '' || initialFilters.date_to !== ''
  );
  const [selectedId, setSelectedId] = useState(() => parseRouteRequestId(routeRequestId));
  const [relaunchTarget, setRelaunchTarget] = useState(null);

  useEffect(() => {
    setSelectedId(parseRouteRequestId(routeRequestId));
  }, [routeRequestId]);

  const { data: meData } = useInstitutionMe();
  const myRole = meData?.institution_role;
  const canManage = canManageRequests(myRole);
  const showBillingSwitch = canEditBilling(myRole);
  const canExport = canExportTransports(myRole);

  const [exportDate, setExportDate] = useState(() => new Date().toISOString().split('T')[0]);
  const [exporting, setExporting] = useState(null); // 'pdf' | null

  const handleDailyExport = useCallback(async () => {
    setExporting('pdf');
    try {
      const { rowsCount } = await exportDailyMissionReportsZip(exportDate);
      const count = Number(rowsCount) || 0;
      toast.success(
        count > 0
          ? `${count} rapport${count > 1 ? 's' : ''} de mission téléchargé${count > 1 ? 's' : ''}`
          : 'Archive des rapports de mission générée',
      );
    } catch (err) {
      toast.error(err?.message || "Erreur lors de l'export");
    } finally {
      setExporting(null);
    }
  }, [exportDate]);

  const updateRequestBilling = useUpdateRequestBilling();
  const updateBookingBilling = useUpdateBookingBilling();
  const sendMutation = useSendRequest();

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
    setSelectedId((prev) => {
      const nextId = prev === req.id ? null : req.id;
      navigate(nextId ? `${requestsBasePath}/${nextId}` : requestsBasePath, { replace: true });
      return nextId;
    });
  }, [navigate, requestsBasePath]);

  const handleClosePanel = useCallback(() => {
    setSelectedId(null);
    navigate(requestsBasePath, { replace: true });
  }, [navigate, requestsBasePath]);

  const handleConfirmRelaunch = useCallback(async () => {
    if (!relaunchTarget?.id) return;
    try {
      await sendMutation.mutateAsync({ requestId: relaunchTarget.id, options: {} });
      setRelaunchTarget(null);
      toast.success('Diffusion relancée auprès des transporteurs');
    } catch (err) {
      toast.error(err?.response?.data?.error || 'Erreur lors de la relance');
    }
  }, [relaunchTarget, sendMutation]);

  // Group by date
  const grouped = useMemo(() => {
    const groups = {};
    for (const req of requests) {
      const label = getDateGroupLabel(req.mission_date || req.scheduled_time);
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
            <div className={`${s.filterGroup} ${s.filterGroupStatus}`}>
              <label htmlFor="filter-status">Statut</label>
              <ChipSelect
                id="filter-status"
                className={s.filterChipSelect}
                options={STATUS_FILTER_OPTIONS}
                value={filters.status}
                onChange={(v) => handleFilter('status', v)}
                placeholder="Tous"
                menuMinWidth={180}
              />
            </div>
            <div className={`${s.filterGroup} ${s.filterGroupMode}`}>
              <label htmlFor="filter-carrier-mode">Mode</label>
              <ChipSelect
                id="filter-carrier-mode"
                className={s.filterChipSelect}
                options={CARRIER_MODE_FILTER_OPTIONS}
                value={filters.carrier_source}
                onChange={(v) => handleFilter('carrier_source', v)}
                placeholder="Tous"
                menuMinWidth={140}
              />
            </div>
            <div className={s.filterDateRange}>
              <div className={s.filterGroup}>
                <label htmlFor="filter-date-from">Du</label>
                <div className={s.filterDateField}>
                  <InlineDatePicker
                    inputId="filter-date-from"
                    value={filters.date_from}
                    onChange={(v) => handleFilter('date_from', v)}
                    ariaLabel="Date de début"
                  />
                </div>
              </div>
              <span className={s.filterDateSep} aria-hidden="true">—</span>
              <div className={s.filterGroup}>
                <label htmlFor="filter-date-to">Au</label>
                <div className={s.filterDateField}>
                  <InlineDatePicker
                    inputId="filter-date-to"
                    value={filters.date_to}
                    onChange={(v) => handleFilter('date_to', v)}
                    ariaLabel="Date de fin"
                  />
                </div>
              </div>
            </div>
            <button
              type="button"
              className={s.clearBtn}
              onClick={() => setFilters({
                status: '',
                carrier_source: '',
                date_from: '',
                date_to: '',
                query: '',
                page: 1,
                per_page: 20,
              })}
            >
              Effacer
            </button>
          </div>
        )}

        {/* Barre d'export journalier (admin + facturation + réception) */}
        {canExport && (
          <div className={s.exportBar}>
            <span className={s.exportBarLabel}>Export journalier</span>
            <div className={s.exportDateField}>
              <InlineDatePicker
                value={exportDate}
                onChange={setExportDate}
                ariaLabel="Date d'export journalier"
              />
            </div>
            <button
              type="button"
              onClick={handleDailyExport}
              disabled={exporting !== null}
              className={s.exportBtnPdf}
              title="Télécharger tous les rapports de mission de la date (1 PDF par transport, archive ZIP)"
            >
              <FaFilePdf size={11} /> {exporting === 'pdf' ? 'Export...' : 'Rapports'}
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
            <p>Aucun transport trouvé</p>
            <p className={s.emptyHint}>Modifiez vos filtres ou créez un nouveau transport.</p>
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

                  const companyName = getCarrierDisplay(req).name;
                  const isExternal = isExternalRequest(req);
                  const carrierModeLabel = getCarrierSourceLabel(req);
                  const delay = resolveDelayInfo(req, nowMs);
                  const tripType = resolveTripTypeMeta(req);
                  const routePoints = getRoutePoints(req);
                  const billingMetaLabel = showBillingSwitch ? null : resolveBillingMetaLabel(req);
                  const timeTypeLabel = req.scheduled_time_type === 'arrival' ? 'RDV' : null;
                  const cardMeta = buildCardMeta({
                    req,
                    companyName,
                    carrierModeLabel,
                    isExternal,
                    tripTypeLabel: showBillingSwitch ? null : tripType.label,
                    billingLabel: billingMetaLabel,
                    timeTypeLabel: showBillingSwitch ? null : timeTypeLabel,
                  });
                  const showRelaunch = canManage && canRelaunchInstitutionRequest(req);

                  return (
                    <div
                      key={req.id}
                      className={`${s.requestCard} ${isSelected ? s.requestCardSelected : ''}`}
                      data-tour-id={req.status === 'DRAFT' ? 'institution-request-draft-card' : undefined}
                      onClick={() => handleSelectRequest(req)}
                    >
                      <div className={`${s.cardIndicator} ${s[st.indicatorClass]}`} />

                      {/* Col 1 : nom + statut hiérarchisé */}
                      <div className={s.colLeft}>
                        <span className={s.patientName}>{patientName}</span>
                        <div className={s.statusBlock}>
                          <div className={s.badgeRow}>
                            <span
                              className={`${s.badge} ${s[st.badgeClass]}`}
                              title={st.fullLabel || st.label}
                            >
                              {st.label}
                            </span>
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
                          {cardMeta.carrierLine && (
                            <span className={s.metaCarrier} title={cardMeta.carrierLine}>
                              {cardMeta.carrierLine}
                            </span>
                          )}
                        </div>
                      </div>

                      {/* Col 2 : trajet complet — départ, destinations, retour éventuel */}
                      <div className={s.colCenter}>
                        {routePoints.map((point, index) => {
                          const isFirst = index === 0;
                          return (
                            <div className={s.routeRow} key={`${point.kind}-${index}-${point.address || ''}`}>
                              <span className={s.routeDot} />
                              <span
                                className={`${s.routeText} ${isFirst ? s.routeTextPrimary : ''}`}
                                title={`${point.label}: ${point.address || '—'}`}
                              >
                                {shortAddr(point.address)}
                              </span>
                            </div>
                          );
                        })}
                      </div>

                      {/* Col 3 : date/heure + actions compactes */}
                      <div className={s.colRight}>
                        <MissionScheduleCardTime
                          request={req}
                          title={[tripType.title, formatLegScheduleSummary(req)].filter(Boolean).join(' — ')}
                        />
                        {cardMeta.detailsLine && (
                          <span className={s.metaDetails} title={cardMeta.detailsLine}>
                            {cardMeta.detailsLine}
                          </span>
                        )}
                        {showBillingSwitch && (
                          <div className={s.colRightActions}>
                            {showRelaunch && (
                              <button
                                type="button"
                                className={s.cardRelaunchBtn}
                                title="Relancer la diffusion aux transporteurs"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  setRelaunchTarget(req);
                                }}
                              >
                                <FaRedo size={10} />
                                Relancer
                              </button>
                            )}
                            {(() => {
                              const isConverted = isConvertedLirie(req) && req.booking_summary;
                              const isPatient = isConverted
                                ? req.booking_summary.billed_to_type !== 'clinic'
                                : (req.billing_intent || 'patient') === 'patient';
                              const isInvoiced = isConverted && req.booking_summary.is_invoiced;
                              return (
                                <div
                                  className={`${s.billingSwitch} ${s.billingSwitchCompact} ${isPatient ? s.billPatient : s.billClinic} ${isInvoiced ? s.billLocked : ''}`}
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
                        )}
                        {!showBillingSwitch && showRelaunch && (
                          <div className={s.colRightActions}>
                            <button
                              type="button"
                              className={s.cardRelaunchBtn}
                              title="Relancer la diffusion aux transporteurs"
                              onClick={(e) => {
                                e.stopPropagation();
                                setRelaunchTarget(req);
                              }}
                            >
                              <FaRedo size={10} />
                              Relancer
                            </button>
                          </div>
                        )}
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

      {relaunchTarget && (
        <ConfirmSendModal
          mode="relaunch"
          onClose={() => setRelaunchTarget(null)}
          onConfirm={handleConfirmRelaunch}
          loading={sendMutation.isPending}
        />
      )}
    </div>
  );
};

export default InstitutionRequests;
