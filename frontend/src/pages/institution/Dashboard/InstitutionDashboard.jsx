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
} from 'react-icons/fa';
import {
  useInstitutionMe,
  useInstitutionRequests,
} from '../../../hooks/useInstitutionData';
import DemoInteractiveGuide from '../../../components/demo/DemoInteractiveGuide';
import { getAuthEnv } from '../../../utils/webAuthSession';
import s from './InstitutionDashboard.module.css';

// ─── Status config ──────────────────────────────────────────
const BOOKING_STATUS_MAP = {
  pending: 'En attente',
  confirmed: 'Confirmé',
  assigned: 'Chauffeur assigné',
  en_route: 'En route',
  in_progress: 'En cours',
  completed: 'Terminé',
  cancelled: 'Annulé',
};

const REQUEST_STATUS_CONFIG = {
  DRAFT:     { label: 'Brouillon',  css: 'statusDraft' },
  SENT:      { label: 'Envoyée',    css: 'statusSent' },
  ACCEPTED:  { label: 'Acceptée',   css: 'statusAccepted' },
  CONVERTED: { label: 'Confirmée',  css: 'statusConverted' },
  CANCELLED: { label: 'Annulée',    css: 'statusCancelled' },
  EXPIRED:   { label: 'Expirée',    css: 'statusExpired' },
};

const BOOKING_STATUS_CSS = {
  pending: 'statusSent',
  confirmed: 'statusConverted',
  assigned: 'statusConverted',
  en_route: 'statusInProgress',
  in_progress: 'statusInProgress',
  completed: 'statusCompleted',
  cancelled: 'statusCancelled',
};

// ─── Helpers ────────────────────────────────────────────────
const shortAddr = (addr, max = 30) => {
  if (!addr) return '—';
  return addr.length > max ? addr.substring(0, max) + '...' : addr;
};

const fmtDate = (d) => {
  if (!d) return '—';
  return new Date(d).toLocaleDateString('fr-CH', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
  });
};

const fmtTime = (d) => {
  if (!d) return '';
  return new Date(d).toLocaleTimeString('fr-CH', {
    hour: '2-digit',
    minute: '2-digit',
  });
};

const resolveDisplayStatus = (req) => {
  if (req.status === 'CONVERTED' && req.booking_summary?.status) {
    const bStatus = req.booking_summary.status.toLowerCase();
    const label = BOOKING_STATUS_MAP[bStatus] || 'Confirmée';
    const css = BOOKING_STATUS_CSS[bStatus] || 'statusConverted';
    return { label, css };
  }
  const cfg = REQUEST_STATUS_CONFIG[req.status] || REQUEST_STATUS_CONFIG.DRAFT;
  return { label: cfg.label, css: cfg.css };
};

const isToday = (dateStr) => {
  if (!dateStr) return false;
  const d = new Date(dateStr);
  const now = new Date();
  return (
    d.getDate() === now.getDate() &&
    d.getMonth() === now.getMonth() &&
    d.getFullYear() === now.getFullYear()
  );
};

// ─── Component ──────────────────────────────────────────────
const InstitutionDashboard = () => {
  const { public_id } = useParams();
  const navigate = useNavigate();
  const location = useLocation();
  const isDemoEnv = getAuthEnv() === 'demo';
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
  const stats = useMemo(() => {
    const items = requestItems;
    const sent = items.filter((r) => r.status === 'SENT').length;

    const activeBookings = items.filter(
      (r) =>
        r.status === 'CONVERTED' &&
        r.booking_summary?.status &&
        !['completed', 'cancelled'].includes(r.booking_summary.status.toLowerCase())
    ).length;

    const completedBookings = items.filter(
      (r) =>
        r.status === 'CONVERTED' &&
        r.booking_summary?.status?.toLowerCase() === 'completed'
    ).length;

    const needsAttention = items.filter((r) => {
      if (r.status !== 'SENT') return false;
      const sentTime = new Date(r.updated_at || r.created_at);
      const hoursAgo = (Date.now() - sentTime.getTime()) / 3600000;
      return hoursAgo > 2;
    }).length;

    return {
      total: requestsData?.total || items.length,
      sent,
      activeBookings,
      completedBookings,
      needsAttention,
    };
  }, [requestItems, requestsData]);

  // ─── Today's requests ───────────────────────────────────
  const todayRequests = useMemo(
    () =>
      requestItems
        .filter((r) => isToday(r.scheduled_time))
        .sort((a, b) => new Date(a.scheduled_time) - new Date(b.scheduled_time)),
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
          to={`/dashboard/institution/${public_id}/requests`}
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
            <div className={s.kpiValue}>{loadingRequests ? '—' : stats.activeBookings}</div>
            <div className={s.kpiLabel}>Transports en cours</div>
          </div>
        </div>

        <div className={`${s.kpiCard} ${s.kpiPending}`} data-tour-id="institution-kpi-pending">
          <div className={s.kpiIconWrap}>
            <FaHourglassHalf />
          </div>
          <div className={s.kpiBody}>
            <div className={s.kpiValue}>{loadingRequests ? '—' : stats.sent}</div>
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
            <div className={s.kpiValue}>{loadingRequests ? '—' : stats.completedBookings}</div>
            <div className={s.kpiLabel}>Terminés</div>
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
              to={`/dashboard/institution/${public_id}/requests?day=today`}
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
                  return (
                    <button
                      key={req.id}
                      className={s.requestItem}
                      onClick={() =>
                        navigate(
                          `/dashboard/institution/${public_id}/requests/${req.id}`
                        )
                      }
                    >
                      <div className={s.reqTime}>
                        <span className={s.reqHour}>{fmtTime(req.scheduled_time)}</span>
                      </div>
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
              to={`/dashboard/institution/${public_id}/requests`}
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
                  return (
                    <button
                      key={req.id}
                      className={s.requestItem}
                      onClick={() =>
                        navigate(
                          `/dashboard/institution/${public_id}/requests/${req.id}`
                        )
                      }
                    >
                      <div className={s.reqTime}>
                        <span className={s.reqDate}>{fmtDate(req.scheduled_time)}</span>
                        <span className={s.reqHour}>{fmtTime(req.scheduled_time)}</span>
                      </div>
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
