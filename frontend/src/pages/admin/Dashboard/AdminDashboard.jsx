import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { FaExclamationTriangle, FaRedoAlt } from 'react-icons/fa';
import { FiArrowRight } from 'react-icons/fi';
import { fetchAdminDashboardSummary } from '../../../services/adminService';
import useAuthToken from '../../../hooks/useAuthToken';
import { adminPaths } from '../routing/adminRoutePaths';
import AdminAttentionCard from './components/AdminAttentionCard';
import AdminMetric from './components/AdminMetric';
import AdminHealthStatus from './components/AdminHealthStatus';
import AdminRecentActivity from './components/AdminRecentActivity';
import styles from './AdminDashboard.module.css';
import shell from '../adminShell.module.css';

const formatUpdatedAt = (iso) => {
  if (!iso) return null;
  try {
    return new Date(iso).toLocaleTimeString('fr-CH', {
      hour: '2-digit',
      minute: '2-digit',
    });
  } catch {
    return null;
  }
};

const formatMoneyChf = (v) => {
  if (v === undefined || v === null) return '—';
  return new Intl.NumberFormat('fr-CH', {
    style: 'currency',
    currency: 'CHF',
    maximumFractionDigits: 0,
  }).format(Number(v));
};

const formatCancelMetric = (count, rate) => {
  const n = Number(count) || 0;
  const pct = (Number(rate) || 0) * 100;
  const pctLabel = pct.toLocaleString('fr-CH', {
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
  });
  return `${n} · ${pctLabel} %`;
};

const AdminDashboard = () => {
  const { public_id: adminId } = useParams();
  const user = useAuthToken();

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

  const greetingName = useMemo(() => {
    const first = String(user?.first_name || '').trim();
    if (first) return first;
    const username = String(user?.username || '').trim();
    if (username) return username;
    return null;
  }, [user]);

  const dataReady = Boolean(summary) && !error;
  const priorities = summary?.priorities;
  const kpi = summary?.kpi_business;
  const plat = summary?.platform_snippet;
  const activity = Array.isArray(summary?.recent_activity)
    ? summary.recent_activity.slice(0, 5)
    : [];

  const bookingsPending = Number(priorities?.bookings_pending_action) || 0;
  const billingReview = Number(priorities?.billing_to_review) || 0;
  const drift = Number(plat?.tenants_in_drift ?? priorities?.tenants_in_drift) || 0;
  const platformActions = Number(plat?.open_alerts ?? priorities?.platform_alerts_open) || 0;
  const critical =
    Number(priorities?.critical_attention_count ?? plat?.critical_attention_count) ||
    drift + platformActions;
  const demosOpen = Number(priorities?.demo_requests_open) || 0;

  const healthStatus = useMemo(() => {
    if (loading && !summary) return 'loading';
    if (error || !summary) return 'unknown';
    const overall = plat?.overall_status;
    if (overall === 'degraded' || critical > 0) return 'degraded';
    if (overall === 'ok') return 'ok';
    return 'unknown';
  }, [loading, summary, error, plat?.overall_status, critical]);

  const healthDetail = useMemo(() => {
    if (healthStatus !== 'degraded') return null;
    const parts = [];
    if (drift > 0) {
      parts.push(
        `${drift} organisation${drift > 1 ? 's' : ''} en dérive`
      );
    }
    if (platformActions > 0) {
      parts.push(
        `${platformActions} action${platformActions > 1 ? 's' : ''} plateforme`
      );
    }
    if (parts.length === 0) return `${critical} élément${critical > 1 ? 's' : ''} à vérifier`;
    return parts.join(' · ');
  }, [healthStatus, drift, platformActions, critical]);

  const criticalExplanation = useMemo(() => {
    if (critical === 0) return 'Situation normale';
    const parts = [];
    if (drift > 0) {
      parts.push(
        `${drift} organisation${drift > 1 ? 's' : ''} en dérive`
      );
    }
    if (platformActions > 0) {
      parts.push(
        `${platformActions} action${platformActions > 1 ? 's' : ''} plateforme`
      );
    }
    return parts.join(' · ') || `${critical} élément${critical > 1 ? 's' : ''} à vérifier`;
  }, [critical, drift, platformActions]);

  const updatedLabel = formatUpdatedAt(summary?.generated_at);
  const fmt = (v) => {
    if (loading && summary == null) return '…';
    if (!dataReady) return '—';
    if (v === undefined || v === null) return '—';
    return v;
  };

  const resolveActivityTo = (item) => {
    if (item?.action === 'open_booking' && item.entity_id != null) {
      return adminPaths.operationsBooking(adminId, item.entity_id);
    }
    return null;
  };

  return (
    <main className={shell.content} data-testid="admin-dashboard">
      <header className={styles.pageHeader}>
        <div className={styles.headerText}>
          <h2 className={styles.greeting}>
            {greetingName ? `Bonjour ${greetingName}` : 'Bonjour'}
          </h2>
          <p className={styles.lead}>
            Voici les éléments nécessitant votre attention.
          </p>
          {updatedLabel ? (
            <p className={styles.updatedAt} data-testid="admin-dash-updated">
              Mis à jour à {updatedLabel}
            </p>
          ) : null}
        </div>
        <Link
          to={adminPaths.operationsBookings(adminId)}
          className={styles.primaryCta}
          data-tour-id="admin-reservations-cta"
        >
          Voir les transports
          <FiArrowRight size={15} aria-hidden />
        </Link>
      </header>

      {error ? (
        <div className={styles.errorBanner} role="alert">
          <FaExclamationTriangle className={styles.errorBannerIcon} aria-hidden />
          <div className={styles.errorBannerBody}>
            <strong className={styles.errorBannerTitle}>Données indisponibles</strong>
            <p className={styles.errorBannerText}>{error}</p>
          </div>
          <button
            type="button"
            className={styles.retryButton}
            onClick={() => load()}
            disabled={loading}
          >
            <FaRedoAlt aria-hidden />
            Réessayer
          </button>
        </div>
      ) : null}

      <section className={styles.block} aria-labelledby="admin-dash-attention-title">
        <h2 id="admin-dash-attention-title" className={styles.blockTitle}>
          À traiter
        </h2>
        <div className={styles.attentionGrid} data-testid="admin-dash-attention">
          <AdminAttentionCard
            title="Transports"
            value={
              dataReady
                ? bookingsPending > 0
                  ? `${bookingsPending} à traiter`
                  : 'Aucun'
                : fmt(bookingsPending)
            }
            explanation={
              bookingsPending > 0
                ? 'Demandes en attente ou sans action'
                : 'Aucune demande en attente'
            }
            to={adminPaths.operationsBookings(adminId)}
            variant={bookingsPending > 0 ? 'attention' : 'ok'}
            linkLabel="Voir les demandes"
          />
          <AdminAttentionCard
            title="Facturation"
            value={
              dataReady
                ? billingReview > 0
                  ? `${billingReview} à contrôler`
                  : 'Aucun'
                : fmt(billingReview)
            }
            explanation="Relevés à vérifier ou valider"
            to={adminPaths.financeReleves(adminId)}
            variant={billingReview > 0 ? 'attention' : 'ok'}
            linkLabel="Voir les relevés"
          />
          <AdminAttentionCard
            title="Alertes plateforme"
            value={
              dataReady
                ? critical > 0
                  ? `${critical} à vérifier`
                  : 'Aucune alerte'
                : fmt(critical)
            }
            explanation={
              critical > 0 ? criticalExplanation : 'Situation normale'
            }
            to={adminPaths.advancedPlatform(adminId, 'overview')}
            variant={critical > 0 ? 'danger' : 'ok'}
            linkLabel={critical > 0 ? 'Voir les détails' : 'Vue plateforme'}
          />
        </div>
        {demosOpen > 0 ? (
          <p className={styles.demoLine} data-testid="admin-dash-demo-line">
            <Link to={`${adminPaths.partnersDemoRequests(adminId)}?status=new`}>
              {demosOpen} nouvelle{demosOpen > 1 ? 's' : ''} demande
              {demosOpen > 1 ? 's' : ''} de démonstration
            </Link>
          </p>
        ) : null}
      </section>

      <section className={styles.block} aria-labelledby="admin-dash-metrics-title">
        <h2 id="admin-dash-metrics-title" className={styles.blockTitle}>
          Activité des 7 derniers jours
        </h2>
        <div className={styles.metricsGrid} data-testid="admin-dash-metrics">
          <AdminMetric label="Créés" value={fmt(kpi?.bookings_created_7d)} />
          <AdminMetric label="Terminés" value={fmt(kpi?.bookings_completed_7d)} />
          <AdminMetric
            label="Annulations"
            value={
              dataReady
                ? formatCancelMetric(
                    kpi?.bookings_canceled_from_created_7d,
                    kpi?.cancellation_rate_7d
                  )
                : '—'
            }
          />
          <AdminMetric
            label="Facturé LIRIE"
            value={
              dataReady
                ? formatMoneyChf(kpi?.platform_invoiced_current_month_chf)
                : '—'
            }
            hint="Ce mois"
          />
        </div>
      </section>

      <div className={styles.healthWrap} data-testid="admin-dash-health">
        <AdminHealthStatus
          status={healthStatus}
          detail={healthDetail}
          detailsTo={adminPaths.advancedPlatform(adminId, 'overview')}
        />
      </div>

      <AdminRecentActivity
        items={dataReady ? activity : []}
        resolveItemTo={resolveActivityTo}
        listTo={adminPaths.operationsBookings(adminId)}
        emptyLabel={
          loading && !summary
            ? 'Chargement de l’activité…'
            : 'Aucun événement récent à afficher.'
        }
      />
    </main>
  );
};

export default AdminDashboard;
