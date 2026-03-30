import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import {
  FaCar,
  FaChartBar,
  FaChevronRight,
  FaClipboardList,
  FaExclamationTriangle,
  FaFileInvoice,
  FaRedoAlt,
  FaServer,
  FaUser,
  FaUserClock,
} from 'react-icons/fa';
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
        title: 'Réservations à traiter',
        subtitle: 'Non terminées (opérationnel)',
        to: `${b}/reservations`,
        value: summary?.priorities?.bookings_pending_action ?? 0,
      },
      {
        key: 'demo_requests_open',
        title: 'Demandes démo non traitées',
        subtitle: 'Statut « nouvelle »',
        to: `${b}/demo-requests?status=new`,
        value: summary?.priorities?.demo_requests_open ?? 0,
      },
      {
        key: 'tenants_suspended',
        title: 'Tenants suspendus',
        subtitle: 'Gouvernance plateforme',
        to: `${b}/platform-ops/tenants`,
        value: summary?.priorities?.tenants_suspended ?? 0,
      },
      {
        key: 'platform_alerts_open',
        title: 'Opérations gouvernance ouvertes',
        subtitle: 'Change requests en cours',
        to: `${b}/platform-ops/overview`,
        value: summary?.priorities?.platform_alerts_open ?? 0,
      },
    ];
  }, [adminId, summary]);

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
    <main className={styles.content}>
      <header className={styles.pageHeader}>
        <div className={styles.pageHeaderText}>
          <div className={styles.titleRow}>
            <h1>Tableau de bord administrateur</h1>
            {loading ? <span className={styles.liveBadge}>Chargement…</span> : null}
            {dataReady ? <span className={styles.liveBadgeOk}>À jour</span> : null}
          </div>
          <p className={styles.pageSub}>
            Synthèse opérationnelle : priorités, état de la plateforme et indicateurs clés.
          </p>
        </div>
        <nav className={styles.quickActions} aria-label="Raccourcis admin">
          <Link to={`${base}/reservations`} className={styles.actionButton}>
            Réservations
          </Link>
          <Link to={`${base}/users`} className={styles.actionButtonGhost}>
            Utilisateurs
          </Link>
          <Link to={`${base}/platform-ops/overview`} className={styles.actionButtonGhost}>
            Plateforme
          </Link>
          <Link to={`${base}/invoices`} className={styles.actionButtonGhost}>
            Factures
          </Link>
          <Link to={`${base}/demo-requests`} className={styles.actionButtonGhost}>
            Demandes démo
          </Link>
        </nav>
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

      <section className={styles.sectionPanel} aria-label="Priorités du jour">
        <div className={styles.sectionHead}>
          <h2 className={styles.sectionHeading}>Priorités du jour</h2>
          <p className={styles.sectionHint}>Accès direct aux écrans concernés.</p>
        </div>
        <div className={styles.priorityGrid}>
          {priorityCards.map((card) => (
            <Link
              key={card.key}
              to={card.to}
              className={`${styles.priorityCardLink} ${
                dataReady && card.value > 0 ? styles.priorityCardAttention : ''
              }`}
            >
              <div className={styles.priorityCardTop}>
                <span className={styles.priorityCardTitle}>{card.title}</span>
                <FaChevronRight className={styles.priorityChevron} aria-hidden />
              </div>
              <span className={styles.priorityCardSubtitle}>{card.subtitle}</span>
              <strong className={styles.priorityCardValue}>{fmtNum(card.value)}</strong>
            </Link>
          ))}
        </div>
      </section>

      <section className={styles.sectionPanel} aria-label="Santé plateforme">
        <div className={styles.sectionHead}>
          <h2 className={styles.sectionHeading}>Santé plateforme</h2>
          <p className={styles.sectionHint}>Aperçu ; le détail se gère dans la console Plateforme.</p>
        </div>
        <div className={styles.platformStrip}>
          <div className={styles.platformStripMain}>
            <div className={styles.platformIconWrap}>
              <FaServer className={styles.platformIcon} aria-hidden />
            </div>
            <div className={styles.platformStatusBlock}>
              <span className={styles.platformLabel}>État global</span>
              <span
                className={
                  !dataReady
                    ? styles.statusPillMuted
                    : plat?.overall_status === 'ok'
                      ? styles.statusPillOk
                      : styles.statusPillWarn
                }
              >
                {!dataReady ? '—' : plat?.overall_status === 'ok' ? 'OK' : 'Dégradé'}
              </span>
            </div>
            <div className={styles.platformCounts}>
              <span className={styles.platformChip}>
                <span className={styles.platformChipLabel}>Alertes</span>
                <span className={styles.platformChipVal}>{fmtNum(plat?.open_alerts)}</span>
              </span>
              <span className={styles.platformChip}>
                <span className={styles.platformChipLabel}>Runbooks (jour)</span>
                <span className={styles.platformChipVal}>{fmtNum(plat?.runbooks_today)}</span>
              </span>
              <span className={styles.platformChip}>
                <span className={styles.platformChipLabel}>Dérive</span>
                <span className={styles.platformChipVal}>{fmtNum(plat?.tenants_in_drift)}</span>
              </span>
            </div>
          </div>
          <div className={styles.platformQuickLinks}>
            <Link to={`${base}/platform-ops/audit`}>Audit</Link>
            <Link to={`${base}/platform-ops/tenants`}>Tenants</Link>
            <Link to={`${base}/platform-ops/runbooks`}>Runbooks</Link>
            <Link to={`${base}/platform-ops/reconciliation`}>Réconciliation</Link>
          </div>
        </div>
      </section>

      <section className={styles.sectionPanel} aria-label="Indicateurs métier">
        <div className={styles.sectionHead}>
          <h2 className={styles.sectionHeading}>Indicateurs métier</h2>
          <p className={styles.sectionHint}>Fenêtres glissantes (7 j. / 30 j.) ou mois civil selon l&apos;indicateur.</p>
        </div>
        <div className={styles.stats}>
          <div className={styles.card}>
            <FaCar className={styles.icon} />
            <div className={styles.cardContent}>
              <h3>Réservations créées (7 jours)</h3>
              <p>{fmtNum(kpi?.bookings_created_7d)}</p>
            </div>
          </div>
          <div className={styles.card}>
            <FaClipboardList className={styles.icon} />
            <div className={styles.cardContent}>
              <h3>Réservations terminées (7 jours)</h3>
              <p>{fmtNum(kpi?.bookings_completed_7d)}</p>
            </div>
          </div>
          <div className={styles.card}>
            <FaChartBar className={styles.icon} />
            <div className={styles.cardContent}>
              <h3>Réservations annulées (7 jours)</h3>
              <p>{fmtNum(kpi?.bookings_canceled_7d)}</p>
            </div>
          </div>
          <div className={styles.card}>
            <FaUserClock className={styles.icon} />
            <div className={styles.cardContent}>
              <h3>Utilisateurs actifs (30 jours, hors admin)</h3>
              <p>{fmtNum(kpi?.active_users_30d)}</p>
            </div>
          </div>
          <div className={styles.card}>
            <FaFileInvoice className={styles.icon} />
            <div className={styles.cardContent}>
              <h3>Factures émises (mois en cours)</h3>
              <p>{fmtNum(kpi?.invoices_current_month)}</p>
            </div>
          </div>
          <div className={styles.card}>
            <FaUser className={styles.icon} />
            <div className={styles.cardContent}>
              <h3>Revenu facturé (mois en cours)</h3>
              <p className={styles.moneyValue}>{fmtMoney(kpi?.revenue_current_month_chf)}</p>
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
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" vertical={false} />
                <XAxis
                  dataKey="month"
                  tick={{ fill: '#64748B', fontSize: 11 }}
                  tickLine={false}
                  axisLine={{ stroke: '#E2E8F0' }}
                />
                <YAxis
                  tick={{ fill: '#64748B', fontSize: 11 }}
                  tickLine={false}
                  axisLine={false}
                  width={40}
                />
                <Tooltip
                  contentStyle={{
                    border: '1px solid #E2E8F0',
                    borderRadius: '8px',
                    fontSize: '12px',
                  }}
                />
                <Legend wrapperStyle={{ fontSize: '12px', color: '#64748B' }} />
                <Line
                  type="monotone"
                  dataKey="bookings"
                  name="Créations"
                  stroke="#00796B"
                  strokeWidth={2}
                  dot={{ r: 3, fill: '#fff', stroke: '#00796B', strokeWidth: 2 }}
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
