// pages/institution/Dashboard/InstitutionDashboard.jsx
/**
 * Centre de pilotage du portail Institution.
 *
 * - Header contextuel (salutation + date)
 * - KPI cards (indicateurs clés)
 * - Transports du jour
 * - Demandes récentes
 *
 * Pas de bouton "Nouvelle demande" (déjà dans le layout header).
 * Pas de bloc "Actions rapides" (déjà dans la sidebar).
 * Pas de bloc "Activité récente" (déjà dans la cloche NotificationBell).
 */

import React, { useMemo } from 'react';
import { useParams, Link, useNavigate, useLocation } from 'react-router-dom';
import {
  FaClipboardList,
  FaArrowRight,
  FaCalendarDay,
  FaMapMarkerAlt,
  FaTruck,
  FaExclamationTriangle,
  FaCheckCircle,
  FaHourglassHalf,
  FaFileAlt,
  FaChevronRight,
  FaTimesCircle,
} from 'react-icons/fa';
import {
  useInstitutionMe,
  useInstitutionRequests,
} from '../../../hooks/useInstitutionData';
import DemoInteractiveGuide from '../../../components/demo/DemoInteractiveGuide';
import { getAuthEnv } from '../../../utils/webAuthSession';
import { computeInstitutionRequestStats, resolveBookingStatusKey } from '../../../utils/institutionBookingStatus';
import { isConvertedLirie } from '../../../utils/requestStatus';
import { BOOKING_STATUS_LABELS } from '../Requests/statusColors';
import {
  getMissionScheduleCardDisplay,
  getNextConfirmedLegTime,
} from '../../../utils/formatLegTime';
import { extractWallClockDate } from '../../../utils/missionTimeDisplay';
import s from './InstitutionDashboard.module.css';

// ─── Status config ──────────────────────────────────────────
const REQUEST_STATUS_CONFIG = {
  DRAFT:     { label: 'Brouillon',  css: 'statusDraft' },
  SENT:      { label: 'Envoyée',    css: 'statusSent' },
  ACCEPTED:  { label: 'Acceptée',   css: 'statusAccepted' },
  CONVERTED: { label: 'Confirmée',  css: 'statusConverted' },
  CANCELLED: { label: 'Annulée',    css: 'statusCancelled' },
  EXPIRED:   { label: 'Expirée',    css: 'statusExpired' },
};

const BOOKING_STATUS_CSS = {
  PENDING: 'statusSent',
  ACCEPTED: 'statusConverted',
  ASSIGNED: 'statusConverted',
  EN_ROUTE: 'statusInProgress',
  IN_PROGRESS: 'statusInProgress',
  OUTBOUND_COMPLETED: 'statusInProgress',
  COMPLETED: 'statusCompleted',
  RETURN_COMPLETED: 'statusCompleted',
  CANCELED: 'statusCancelled',
};

// ─── Helpers ────────────────────────────────────────────────
const shortAddr = (addr, max = 30) => {
  if (!addr) return '—';
  return addr.length > max ? addr.substring(0, max) + '...' : addr;
};

const fmtDate = (d) => {
  if (!d) return '—';
  const raw = String(d).trim();
  const iso = /^\d{4}-\d{2}-\d{2}$/.test(raw) ? `${raw}T12:00:00` : raw;
  const parsed = new Date(iso);
  if (Number.isNaN(parsed.getTime())) return '—';
  return parsed.toLocaleDateString('fr-CH', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
  });
};

/** Date + heure mission (mission_date, legs RDV, départ confirmé). */
const resolveDashboardSchedule = (req) => {
  const { primary } = getMissionScheduleCardDisplay(req);
  const dateSrc = req?.mission_date
    || extractWallClockDate(req?.scheduled_time)
    || extractWallClockDate(req?.next_confirmed_time)
    || extractWallClockDate(getNextConfirmedLegTime(req)?.iso);
  const date = dateSrc ? fmtDate(dateSrc) : '—';
  const hour = primary?.time || '';
  const hourKind = primary?.label || '';
  return { date, hour, hourKind };
};

const getRequestMissionDate = (req) => (
  req?.mission_date
  || extractWallClockDate(req?.scheduled_time)
  || extractWallClockDate(req?.next_confirmed_time)
  || extractWallClockDate(getNextConfirmedLegTime(req)?.iso)
  || null
);

const getRequestSortTime = (req) => {
  const next = getNextConfirmedLegTime(req);
  const src = next?.iso || req?.next_confirmed_time || req?.scheduled_time || req?.mission_date;
  if (!src) return 0;
  const raw = String(src).trim();
  const iso = /^\d{4}-\d{2}-\d{2}$/.test(raw) ? `${raw}T00:00:00` : raw;
  const parsed = new Date(iso);
  return Number.isNaN(parsed.getTime()) ? 0 : parsed.getTime();
};

const resolveDisplayStatus = (req) => {
  if (isConvertedLirie(req) && req.booking_summary?.status) {
    const bookingKey = resolveBookingStatusKey(req.booking_summary);
    const label = BOOKING_STATUS_LABELS[bookingKey] || 'Confirmée';
    const css = BOOKING_STATUS_CSS[bookingKey] || 'statusConverted';
    return { label, css };
  }
  const cfg = REQUEST_STATUS_CONFIG[req.status] || REQUEST_STATUS_CONFIG.DRAFT;
  return { label: cfg.label, css: cfg.css };
};

const isToday = (dateStr) => {
  if (!dateStr) return false;
  const raw = String(dateStr).trim();
  const iso = /^\d{4}-\d{2}-\d{2}$/.test(raw) ? `${raw}T12:00:00` : raw;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return false;
  const now = new Date();
  return (
    d.getDate() === now.getDate() &&
    d.getMonth() === now.getMonth() &&
    d.getFullYear() === now.getFullYear()
  );
};

const isTodayRequest = (req) => isToday(getRequestMissionDate(req));

const RequestItemTime = ({ schedule, showDate = false }) => {
  const showKind = Boolean(schedule.hourKind);
  return (
    <div className={`${s.reqTime} ${showDate ? s.reqTimeWithDate : ''}`}>
      {showDate ? <span className={s.reqDate}>{schedule.date}</span> : null}
      <span className={s.reqHourLine}>
        <span className={s.reqHour}>{schedule.hour || '—'}</span>
        <span className={s.reqHourKind}>{showKind ? schedule.hourKind : '\u00a0'}</span>
      </span>
    </div>
  );
};

// ─── Component ──────────────────────────────────────────────
const InstitutionDashboard = () => {
  const { public_id } = useParams();
  const navigate = useNavigate();
  const location = useLocation();
  const isDemoEnv = getAuthEnv() === 'demo';
  const requestsBasePath = useMemo(() => {
    const isDemoPath = String(location.pathname || '').startsWith('/demo/');
    const dashboardRoot = isDemoPath ? '/demo/dashboard' : '/dashboard';
    return `${dashboardRoot}/institution/${public_id}/requests`;
  }, [location.pathname, public_id]);
  const fallbackDemoMission = isDemoEnv
    ? (
        localStorage.getItem('demo_recommended_journey') ||
        localStorage.getItem('demo_demo_recommended_journey') ||
        ''
      )
        .toString()
        .trim()
        .toLowerCase()
    : '';
  const isInstitutionDemoCompleted = (() => {
    try {
      return window.sessionStorage.getItem('demo_institution_journey_completed') === '1';
    } catch {
      return false;
    }
  })();
  const demoMission = useMemo(() => {
    if (isInstitutionDemoCompleted) return null;
    const mission = new URLSearchParams(location.search).get('demo_mission');
    if (mission) return mission;
    if (fallbackDemoMission === 'institution') return 'institution';
    return null;
  }, [location.search, fallbackDemoMission, isInstitutionDemoCompleted]);

  const { data: institutionData } = useInstitutionMe();
  const { data: requestsData, isLoading: loadingRequests } = useInstitutionRequests({
    per_page: 20,
  });

  const user = institutionData?.user;

  const requestItems = useMemo(
    () => requestsData?.requests || requestsData?.items || [],
    [requestsData]
  );

  // ─── Stats ──────────────────────────────────────────────
  const stats = useMemo(
    () => computeInstitutionRequestStats(
      requestItems,
      requestsData?.total || requestItems.length,
    ),
    [requestItems, requestsData],
  );

  // ─── Today's requests ───────────────────────────────────
  const todayRequests = useMemo(
    () =>
      requestItems
        .filter((r) => isTodayRequest(r))
        .sort((a, b) => getRequestSortTime(a) - getRequestSortTime(b)),
    [requestItems]
  );

  // ─── Recent requests (latest 5, excluding today to avoid overlap) ──
  const recentRequests = useMemo(
    () =>
      [...requestItems]
        .sort((a, b) => new Date(b.created_at) - new Date(a.created_at))
        .slice(0, 5),
    [requestItems]
  );

  // ─── Date display ──────────────────────────────────────
  const now = new Date();
  const todayStr = now.toLocaleDateString('fr-CH', {
    weekday: 'long',
    day: 'numeric',
    month: 'long',
    year: 'numeric',
  });

  const greeting = (() => {
    const h = now.getHours();
    if (h < 12) return 'Bonjour';
    if (h < 18) return 'Bon après-midi';
    return 'Bonsoir';
  })();

  // Afficher le prénom uniquement s'il est renseigné (pas le fallback email)
  const userName = user?.first_name || '';

  return (
    <div className={s.dashboard}>
      {demoMission === 'institution' && (
        <DemoInteractiveGuide
          role="institution"
          onFinish={() => {}}
          userFirstName={user?.first_name}
        />
      )}
      {/* ── Header contextuel (pas de bouton, déjà dans le layout) ── */}
      <div className={s.dashHeader} data-tour-id="institution-dashboard">
        <h2 className={s.dashTitle}>
          {greeting}{userName ? ` ${userName}` : ''}
        </h2>
        <div className={s.dashMeta}>
          <FaCalendarDay className={s.dashMetaIcon} />
          <span className={s.dashDate}>{todayStr}</span>
          {todayRequests.length > 0 && (
            <span className={s.dashDaySummary}>
              {todayRequests.length} transport{todayRequests.length > 1 ? 's' : ''} aujourd'hui
            </span>
          )}
        </div>
      </div>

      {/* ── KPI Cards ─────────────────────────────────────── */}
      <div className={s.kpiGrid} data-tour-id="institution-kpi-grid">
        <Link
          to={requestsBasePath}
          className={`${s.kpiCard} ${s.kpiTotal}`}
        >
          <div className={s.kpiIconWrap}>
            <FaClipboardList />
          </div>
          <div className={s.kpiBody}>
            <div className={s.kpiValue}>{loadingRequests ? '—' : stats.total}</div>
            <div className={s.kpiLabel}>Demandes totales</div>
          </div>
        </Link>

        <div className={`${s.kpiCard} ${s.kpiActive}`}>
          <div className={s.kpiIconWrap}>
            <FaTruck />
          </div>
          <div className={s.kpiBody}>
            <div className={s.kpiValue}>{loadingRequests ? '—' : stats.active}</div>
            <div className={s.kpiLabel}>Transports en cours</div>
          </div>
        </div>

        <div className={`${s.kpiCard} ${s.kpiPending}`} data-tour-id="institution-kpi-pending">
          <div className={s.kpiIconWrap}>
            <FaHourglassHalf />
          </div>
          <div className={s.kpiBody}>
            <div className={s.kpiValue}>{loadingRequests ? '—' : stats.pending}</div>
            <div className={s.kpiLabel}>En attente</div>
          </div>
          {stats.needsAttention > 0 && (
            <div className={s.kpiAlert} title={`${stats.needsAttention} demande(s) en attente depuis plus de 2h`}>
              <FaExclamationTriangle /> {stats.needsAttention}
            </div>
          )}
        </div>

        <div className={`${s.kpiCard} ${s.kpiCompleted}`}>
          <div className={s.kpiIconWrap}>
            <FaCheckCircle />
          </div>
          <div className={s.kpiBody}>
            <div className={s.kpiValue}>{loadingRequests ? '—' : stats.completed}</div>
            <div className={s.kpiLabel}>Terminés</div>
          </div>
        </div>

        <div className={`${s.kpiCard} ${s.kpiCancelled}`}>
          <div className={s.kpiIconWrap}>
            <FaTimesCircle />
          </div>
          <div className={s.kpiBody}>
            <div className={s.kpiValue}>{loadingRequests ? '—' : stats.cancelled}</div>
            <div className={s.kpiLabel}>Annulés</div>
          </div>
        </div>
      </div>

      {/* ── Transports du jour + Demandes récentes ────────── */}
      <div className={s.cardsGrid}>
        {/* Transports du jour */}
        <div className={s.card} data-tour-id="institution-create-request">
          <div className={s.cardHeader}>
            <div className={s.cardHeaderLeft}>
              <FaCalendarDay className={s.cardHeaderIcon} />
              <h3 className={s.cardTitle}>Transports du jour</h3>
              {todayRequests.length > 0 && (
                <span className={s.cardCount}>{todayRequests.length}</span>
              )}
            </div>
            <Link
              to={`${requestsBasePath}?day=today`}
              className={s.cardLink}
            >
              Tout voir <FaChevronRight />
            </Link>
          </div>

          <div className={s.cardBody}>
            {loadingRequests ? (
              <div className={s.loadingState}>Chargement...</div>
            ) : todayRequests.length > 0 ? (
              <div className={s.requestList}>
                {todayRequests.map((req) => {
                  const st = resolveDisplayStatus(req);
                  const schedule = resolveDashboardSchedule(req);
                  return (
                    <button
                      key={req.id}
                      className={s.requestItem}
                      onClick={() => navigate(`${requestsBasePath}/${req.id}`)}
                    >
                      <RequestItemTime schedule={schedule} />
                      <div className={s.reqInfo}>
                        <div className={s.reqPatient}>
                          {req.patient
                            ? `${req.patient.last_name} ${req.patient.first_name}`
                            : req.external_reference || '—'}
                        </div>
                        <div className={s.reqRoute}>
                          <FaMapMarkerAlt className={s.reqRouteIcon} />
                          <span>{shortAddr(req.pickup_location)}</span>
                          <FaArrowRight className={s.reqArrow} />
                          <span>{shortAddr(req.dropoff_location)}</span>
                        </div>
                      </div>
                      <span className={`${s.statusBadge} ${s[st.css]}`}>{st.label}</span>
                    </button>
                  );
                })}
              </div>
            ) : (
              <div className={s.emptyState}>
                <FaCalendarDay className={s.emptyIcon} />
                <p>Aucun transport prévu aujourd'hui</p>
              </div>
            )}
          </div>
        </div>

        {/* Demandes récentes */}
        <div className={s.card} data-tour-id="institution-history">
          <div className={s.cardHeader}>
            <div className={s.cardHeaderLeft}>
              <FaFileAlt className={s.cardHeaderIcon} />
              <h3 className={s.cardTitle}>Demandes récentes</h3>
            </div>
            <Link
              to={requestsBasePath}
              className={s.cardLink}
            >
              Tout voir <FaChevronRight />
            </Link>
          </div>

          <div className={s.cardBody}>
            {loadingRequests ? (
              <div className={s.loadingState}>Chargement...</div>
            ) : recentRequests.length > 0 ? (
              <div className={s.requestList}>
                {recentRequests.map((req) => {
                  const st = resolveDisplayStatus(req);
                  const schedule = resolveDashboardSchedule(req);
                  return (
                    <button
                      key={req.id}
                      className={s.requestItem}
                      onClick={() => navigate(`${requestsBasePath}/${req.id}`)}
                    >
                      <RequestItemTime schedule={schedule} showDate />
                      <div className={s.reqInfo}>
                        <div className={s.reqPatient}>
                          {req.patient
                            ? `${req.patient.last_name} ${req.patient.first_name}`
                            : req.external_reference || '—'}
                        </div>
                        <div className={s.reqRoute}>
                          <FaMapMarkerAlt className={s.reqRouteIcon} />
                          <span>{shortAddr(req.pickup_location)}</span>
                          <FaArrowRight className={s.reqArrow} />
                          <span>{shortAddr(req.dropoff_location)}</span>
                        </div>
                      </div>
                      <span className={`${s.statusBadge} ${s[st.css]}`}>{st.label}</span>
                    </button>
                  );
                })}
              </div>
            ) : (
              <div className={s.emptyState}>
                <FaClipboardList className={s.emptyIcon} />
                <p>Aucune demande récente</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default InstitutionDashboard;
