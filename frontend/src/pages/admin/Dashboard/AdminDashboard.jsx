import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import {
  FiActivity,
  FiAlertTriangle,
  FiClipboard,
  FiDollarSign,
  FiFileText,
  FiLayers,
  FiPlus,
  FiServer,
  FiUsers,
  FiXCircle,
  FiZap,
} from 'react-icons/fi';
import { FaExclamationTriangle, FaRedoAlt } from 'react-icons/fa';
import InlineDatePicker from '../../../components/ui/InlineDatePicker';
import ov from '../../company/Dashboard/components/OverviewCards.module.css';
import dms from '../../company/Dashboard/components/DispatchModeStatusBar.module.css';
import { fetchAdminDashboardSummary } from '../../../services/adminService';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import styles from './AdminDashboard.module.css';
import shell from '../adminShell.module.css';

function makeToday() {
  const d = new Date();
  const pad = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}

const formatFrDateTime = (iso) => {
  if (!iso) return '—';
  try {
    return new Date(iso).toLocaleString('fr-FR', {
      day: '2-digit',
      month: '2-digit',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  } catch {
    return String(iso);
  }
};

const statusMeta = (status) => {
  const key = String(status || '').toLowerCase();
  if (['completed', 'done', 'return_completed', 'terminee', 'terminée'].includes(key)) {
    return { label: 'Terminée', className: styles.activityStatusOk };
  }
  if (['pending', 'en_attente'].includes(key)) {
    return { label: 'En attente', className: styles.activityStatusPending };
  }
  if (['assigned', 'accepted', 'in_progress', 'en_route', 'en_cours'].includes(key)) {
    return { label: 'En cours', className: styles.activityStatusProgress };
  }
  if (['canceled', 'cancelled', 'annulee', 'annulée', 'rejected'].includes(key)) {
    return { label: 'Annulée', className: styles.activityStatusDanger };
  }
  return { label: status || '—', className: styles.activityStatusDefault };
};

const AdminDashboard = () => {
  const { public_id: adminId } = useParams();

  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [dashboardDay, setDashboardDay] = useState(makeToday);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchAdminDashboardSummary();
      setSummary(data || null);
    } catch {
      setError('Impossible de charger le résumé. Vérifiez la connexion ou réessayez.');
      setSummary(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [adminId, load]);

  const base = `/dashboard/admin/${adminId}`;

  const priorityCards = useMemo(() => {
    const b = `/dashboard/admin/${adminId}`;
    return [
      {
        key: 'bookings_pending_action',
        title: 'À traiter',
        subtitle: 'Réservations (opérationnel)',
        Icon: FiClipboard,
        to: `${b}/reservations`,
        value: summary?.priorities?.bookings_pending_action ?? 0,
        accentKey: 'bookings_pending_action',
      },
      {
        key: 'demo_requests_open',
        title: 'Démos',
        subtitle: 'Demandes nouvelles',
        Icon: FiLayers,
        to: `${b}/demo-requests?status=new`,
        value: summary?.priorities?.demo_requests_open ?? 0,
        accentKey: 'demo_requests_open',
      },
      {
        key: 'tenants_suspended',
        title: 'Tenants',
        subtitle: 'Suspendus',
        Icon: FiAlertTriangle,
        to: `${b}/platform-ops/tenants`,
        value: summary?.priorities?.tenants_suspended ?? 0,
        accentKey: 'tenants_suspended',
      },
      {
        key: 'platform_alerts_open',
        title: 'Gouvernance',
        subtitle: 'CR ouverts',
        Icon: FiServer,
        to: `${b}/platform-ops/overview`,
        value: summary?.priorities?.platform_alerts_open ?? 0,
        accentKey: 'platform_alerts_open',
      },
    ];
  }, [adminId, summary]);

  const priorityAccent = (card, v) => {
    const n = Number(v) || 0;
    if (card.accentKey === 'tenants_suspended' && n > 0) return 'danger';
    if (n > 0) return 'warning';
    return 'default';
  };

  const kpi = summary?.kpi_business;
  const plat = summary?.platform_snippet;
  const trends = summary?.booking_trends;
  const activity = Array.isArray(summary?.recent_activity) ? summary.recent_activity : [];
  const dataReady = Boolean(summary) && !error;

  const fmtNum = (v) => {
    if (loading && summary == null) return '…';
    if (!dataReady) return '—';
    if (v === undefined || v === null) return '—';
    return v;
  };

  const fmtMoney = (v) => {
    if (loading && summary == null) return '…';
    if (!dataReady) return '—';
    if (v === undefined || v === null) return '—';
    return `${Number(v).toFixed(2)} CHF`;
  };

  return (
    <main className={shell.content}>
      <header className={styles.dashboardHeader}>
        <div className={styles.headerLeft}>
          <h1 className={styles.headerTitle} data-tour-id="admin-dashboard-title">
            Tableau de bord administrateur
          </h1>
          <div className={styles.headerMeta}>
            <InlineDatePicker value={dashboardDay} onChange={(iso) => setDashboardDay(iso)} />
            {loading ? <span className={styles.liveBadge}>Chargement…</span> : null}
            {dataReady ? (
              <span className={styles.liveBadgeOk}>
                <span className={styles.liveDot} aria-hidden />
                À jour
              </span>
            ) : null}
          </div>
        </div>
        <div className={styles.headerActions}>
          <Link to={`${base}/platform-ops/overview`} className={styles.headerBtnSecondary}>
            <FiZap size={16} aria-hidden />
            Plateforme
          </Link>
          <Link to={`${base}/reservations`} className={styles.headerBtnPrimary} data-tour-id="admin-reservations-cta">
            <FiPlus size={16} aria-hidden />
            Réservations
          </Link>
        </div>
      </header>

      {error ? (
        <div className={styles.errorBanner} role="alert">
          <FaExclamationTriangle className={styles.errorBannerIcon} aria-hidden />
          <div className={styles.errorBannerBody}>
            <strong className={styles.errorBannerTitle}>Données indisponibles</strong>
            <p className={styles.errorBannerText}>{error}</p>
          </div>
          <button type="button" className={styles.retryButton} onClick={() => load()} disabled={loading}>
            <FaRedoAlt aria-hidden />
            Réessayer
          </button>
        </div>
      ) : null}

      <div className={ov.kpiGrid} data-tour-id="admin-kpi-grid" aria-label="Priorités">
        {priorityCards.map((card) => {
          const acc = priorityAccent(card, card.value);
          const accentClass = ov[`accent_${acc}`] || '';
          const Icon = card.Icon;
          return (
            <Link
              key={card.key}
              to={card.to}
              className={`${ov.kpiCard} ${accentClass}`}
              title={`${card.title} — ${card.subtitle}`}
            >
              <div className={ov.kpiIconContainer}>
                <Icon className={ov.kpiIcon} aria-hidden />
              </div>
              <div className={ov.kpiContent}>
                <span className={ov.kpiLabel}>{card.title}</span>
                <span className={styles.kpiLinkSub}>{card.subtitle}</span>
                <span className={ov.kpiValue}>{fmtNum(card.value)}</span>
              </div>
            </Link>
          );
        })}
      </div>

      <div
        className={`${dms.bar} ${styles.adminPlatformBar} ${
          !dataReady ? dms.bar_neutral : plat?.overall_status === 'ok' ? dms.bar_brand : dms.bar_neutral
        }`}
        aria-label="Santé plateforme"
      >
        <div className={dms.modeInfo}>
          <FiServer size={14} className={dms.modeIcon} aria-hidden />
          <span className={dms.modeLabel}>Plateforme</span>
          <span className={dms.modeSep}>—</span>
          <span className={dms.modeDesc}>
            {!dataReady
              ? 'Chargement du statut…'
              : plat?.overall_status === 'ok'
                ? 'État global nominal'
                : 'État dégradé — vérifier la console Ops'}
          </span>
        </div>
        <div className={styles.platformBarRight}>
          <span className={dms.aiSuggestions}>
            Alertes {fmtNum(plat?.open_alerts)} · Runbooks {fmtNum(plat?.runbooks_today)} · Dérive{' '}
            {fmtNum(plat?.tenants_in_drift)}
          </span>
          <div className={styles.platformBarLinks}>
            <Link to={`${base}/platform-ops/audit`}>Audit</Link>
            <span className={styles.platformBarSep}>·</span>
            <Link to={`${base}/platform-ops/tenants`}>Tenants</Link>
            <span className={styles.platformBarSep}>·</span>
            <Link to={`${base}/platform-ops/runbooks`}>Runbooks</Link>
            <span className={styles.platformBarSep}>·</span>
            <Link to={`${base}/platform-ops/reconciliation`}>Réconciliation</Link>
          </div>
        </div>
      </div>

      <section className={styles.sectionPanel} aria-label="Indicateurs métier">
        <div className={styles.sectionHead}>
          <h2 className={styles.sectionHeading}>Indicateurs métier</h2>
          <p className={styles.sectionHint}>
            Fenêtres glissantes (7 j. / 30 j.) ou mois civil selon l&apos;indicateur.
          </p>
        </div>
        <div className={`${ov.kpiGrid} ${styles.bizKpiGrid}`}>
          <div className={`${ov.kpiCard} ${ov.accent_default}`}>
            <div className={ov.kpiIconContainer}>
              <FiActivity className={ov.kpiIcon} aria-hidden />
            </div>
            <div className={ov.kpiContent}>
              <span className={ov.kpiLabel}>Créées (7 j.)</span>
              <span className={ov.kpiValue}>{fmtNum(kpi?.bookings_created_7d)}</span>
            </div>
          </div>
          <div className={`${ov.kpiCard} ${ov.accent_brand}`}>
            <div className={ov.kpiIconContainer}>
              <FiClipboard className={ov.kpiIcon} aria-hidden />
            </div>
            <div className={ov.kpiContent}>
              <span className={ov.kpiLabel}>Terminées (7 j.)</span>
              <span className={ov.kpiValue}>{fmtNum(kpi?.bookings_completed_7d)}</span>
            </div>
          </div>
          <div className={`${ov.kpiCard} ${ov.accent_default}`}>
            <div className={ov.kpiIconContainer}>
              <FiXCircle className={ov.kpiIcon} aria-hidden />
            </div>
            <div className={ov.kpiContent}>
              <span className={ov.kpiLabel}>Annulées (7 j.)</span>
              <span className={ov.kpiValue}>{fmtNum(kpi?.bookings_canceled_7d)}</span>
            </div>
          </div>
          <div className={`${ov.kpiCard} ${ov.accent_brand}`}>
            <div className={ov.kpiIconContainer}>
              <FiUsers className={ov.kpiIcon} aria-hidden />
            </div>
            <div className={ov.kpiContent}>
              <span className={ov.kpiLabel}>Utilisateurs actifs (30 j.)</span>
              <span className={ov.kpiValue}>{fmtNum(kpi?.active_users_30d)}</span>
            </div>
          </div>
          <div className={`${ov.kpiCard} ${ov.accent_default}`}>
            <div className={ov.kpiIconContainer}>
              <FiFileText className={ov.kpiIcon} aria-hidden />
            </div>
            <div className={ov.kpiContent}>
              <span className={ov.kpiLabel}>Factures (mois)</span>
              <span className={ov.kpiValue}>{fmtNum(kpi?.invoices_current_month)}</span>
            </div>
          </div>
          <div className={`${ov.kpiCard} ${ov.accent_success}`}>
            <div className={ov.kpiIconContainer}>
              <FiDollarSign className={ov.kpiIcon} aria-hidden />
            </div>
            <div className={ov.kpiContent}>
              <span className={ov.kpiLabel}>Revenu facturé (CHF)</span>
              <span className={ov.kpiValue}>{fmtMoney(kpi?.revenue_current_month_chf)}</span>
            </div>
          </div>
        </div>
      </section>

      <section className={`${styles.sectionPanel} ${styles.chartPanel}`} aria-label="Évolution des réservations">
        <div className={styles.sectionHead}>
          <h2 className={styles.sectionHeading}>Réservations par mois — 12 derniers mois</h2>
          <p className={styles.chartSub}>
            Créations (agrégat mensuel).{' '}
            <Link to={`${base}/reservations`} className={styles.inlineLink}>
              Voir les réservations
            </Link>
          </p>
        </div>
        {trends && trends.length > 0 ? (
          <div className={styles.chartWrap}>
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={trends} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-primary)" vertical={false} />
                <XAxis
                  dataKey="month"
                  tick={{ fill: 'var(--text-tertiary)', fontSize: 11 }}
                  tickLine={false}
                  axisLine={{ stroke: 'var(--border-primary)' }}
                />
                <YAxis
                  tick={{ fill: 'var(--text-tertiary)', fontSize: 11 }}
                  tickLine={false}
                  axisLine={false}
                  width={40}
                />
                <Tooltip
                  contentStyle={{
                    border: '1px solid var(--border-primary)',
                    borderRadius: 'var(--radius-md)',
                    fontSize: 'var(--font-xs)',
                    background: 'var(--bg-primary)',
                    boxShadow: 'var(--shadow-sm)',
                  }}
                  labelStyle={{ color: 'var(--text-secondary)' }}
                />
                <Legend
                  wrapperStyle={{
                    fontSize: 'var(--font-xs)',
                    color: 'var(--text-tertiary)',
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="bookings"
                  name="Créations"
                  stroke="var(--brand-primary)"
                  strokeWidth={2}
                  dot={{
                    r: 3,
                    fill: 'var(--bg-primary)',
                    stroke: 'var(--brand-primary)',
                    strokeWidth: 2,
                  }}
                  activeDot={{ r: 4 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div className={styles.emptyChart}>
            <p>Aucune donnée d&apos;évolution pour l&apos;instant.</p>
          </div>
        )}
      </section>

      <section className={`${styles.sectionPanel} ${styles.activitySection}`} aria-label="Activité récente">
        <div className={styles.sectionHead}>
          <h2 className={styles.sectionHeading}>Activité récente</h2>
          <p className={styles.sectionHint}>Derniers événements (extrait).</p>
        </div>
        {!dataReady && !loading ? (
          <p className={styles.activityEmpty}>Indicateurs non chargés.</p>
        ) : activity.length === 0 ? (
          <p className={styles.activityEmpty}>Aucun événement récent à afficher.</p>
        ) : (
          <ul className={styles.activityList}>
            {activity.map((item, idx) => {
              const st = statusMeta(item.status);
              return (
                <li key={`${item.type}-${item.occurred_at}-${idx}`} className={styles.activityItem}>
                  <span className={styles.activityTime}>{formatFrDateTime(item.occurred_at)}</span>
                  <span className={styles.activityLabel}>{item.label}</span>
                  <span className={`${styles.activityStatus} ${st.className}`}>{st.label}</span>
                </li>
              );
            })}
          </ul>
        )}
      </section>
    </main>
  );
};

export default AdminDashboard;
