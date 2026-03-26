import React, { useCallback, useEffect, useState } from 'react';
import { FaHeartbeat, FaServer } from 'react-icons/fa';
import HeaderDashboard from '../../../components/layout/Header/HeaderDashboard';
import AdminSidebar from '../../../components/layout/Sidebar/AdminSidebar/AdminSidebar';
import StatusBadge from '../../../components/platform/StatusBadge';
import { fetchPlatformStatus } from '../../../services/adminService';
import styles from './AdminPlatformOps.module.css';

const POLL_MS_VISIBLE = 30000;
const POLL_MS_HIDDEN = 120000;

function formatTime(iso) {
  if (!iso) return '—';
  try {
    return new Date(iso).toLocaleString('fr-CH');
  } catch {
    return String(iso);
  }
}

function EnvCard({ title, env }) {
  if (!env) return null;
  const { monitored, status, latency_ms: latencyMs, checks = {}, errors = [] } = env;
  return (
    <div className={styles.card}>
      <h2 className={styles.cardTitle}>
        <FaServer className={styles.cardIcon} aria-hidden />
        {title}
      </h2>
      <div className={styles.cardMeta}>
        <StatusBadge
          status={status}
          title={monitored ? 'Statut agrégé' : 'Environnement non suivi'}
        />
        {monitored && latencyMs != null && (
          <span>· dernière mesure ~ {Math.round(latencyMs)} ms</span>
        )}
        {!monitored && <span>· collecte désactivée (config)</span>}
      </div>
      {monitored && (
        <ul className={styles.checks}>
          {Object.entries(checks).map(([k, v]) => (
            <li key={k}>
              <strong>{k}</strong> : <StatusBadge status={v?.status} />
            </li>
          ))}
        </ul>
      )}
      {errors.length > 0 && (
        <div className={styles.callout}>
          {errors.map((e, i) => (
            <div key={i}>
              {e.type ? `[${e.type}] ` : ''}
              {e.message}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/**
 * Affiche pourquoi l’état reste « Inconnu » : ce n’est pas un bug front — l’API
 * n’agrège des checks que si PLATFORM_API_URL_* est défini côté serveur.
 */
function ConfigHint({ data }) {
  if (!data) return null;
  const prod = data.environments?.prod;
  const demo = data.environments?.demo;
  const links = data.links || {};
  const needProd = !prod?.monitored;
  const needDemo = !demo?.monitored;
  const needObs =
    !links.grafana && !links.prometheus && !links.alertmanager;
  if (!needProd && !needDemo && !needObs) return null;
  return (
    <div className={styles.configHint} role="note">
      <strong>Pourquoi peu de données ?</strong> Cette page appelle le backend ; les checks
      (ready, base, Redis, WebSocket) ne s’affichent que si les URLs cibles sont configurées sur
      le <strong>serveur API</strong> (pas le navigateur). Définir au minimum{' '}
      <code>PLATFORM_API_URL_PROD</code> (et optionnellement <code>PLATFORM_API_URL_DEMO</code>),
      puis <code>PLATFORM_LINK_GRAFANA</code> / <code>PLATFORM_LINK_PROMETHEUS</code> /{' '}
      <code>PLATFORM_LINK_ALERTMANAGER</code> pour les boutons. Voir{' '}
      <code>backend/env.example</code> — redémarrer l’API après modification.
    </div>
  );
}

function ObservabilityLinks({ links }) {
  const items = [
    { key: 'grafana', label: 'Grafana' },
    { key: 'prometheus', label: 'Prometheus' },
    { key: 'alertmanager', label: 'Alertmanager' },
  ];
  return (
    <div className={styles.card}>
      <h2 className={styles.cardTitle}>Observabilité</h2>
      <p className={styles.cardMeta}>Liens externes (pas d’intégration embarquée)</p>
      <div className={styles.linksRow}>
        {items.map(({ key, label }) => {
          const href = links?.[key];
          if (!href) {
            return (
              <button key={key} type="button" className={styles.linkBtn} disabled>
                {label} — non configuré
              </button>
            );
          }
          return (
            <a
              key={key}
              href={href}
              target="_blank"
              rel="noopener noreferrer"
              className={styles.linkBtn}
            >
              Ouvrir {label}
            </a>
          );
        })}
      </div>
    </div>
  );
}

/**
 * Console Admin Ops / Platform — lecture seule (agrégateur backend).
 */
const AdminPlatformOps = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastOk, setLastOk] = useState(null);

  const load = useCallback(async () => {
    setError(null);
    try {
      const json = await fetchPlatformStatus();
      setData(json);
      setLastOk(new Date().toISOString());
    } catch (e) {
      const msg =
        e?.response?.status === 403
          ? 'Accès refusé (403). Vérifiez le rôle admin et la whitelist IP.'
          : e?.response?.data?.message || e?.message || 'Erreur de chargement';
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    let intervalId;
    const reschedule = () => {
      clearInterval(intervalId);
      const ms = document.hidden ? POLL_MS_HIDDEN : POLL_MS_VISIBLE;
      intervalId = setInterval(load, ms);
    };
    load();
    reschedule();
    const onVis = () => {
      reschedule();
      if (!document.hidden) load();
    };
    document.addEventListener('visibilitychange', onVis);
    return () => {
      clearInterval(intervalId);
      document.removeEventListener('visibilitychange', onVis);
    };
  }, [load]);

  return (
    <div className={styles.adminContainer}>
      <HeaderDashboard />
      <div className={styles.dashboard}>
        <AdminSidebar />
        <main className={styles.content}>
          <header className={styles.pageHeader}>
            <div className={styles.pageHeaderText}>
              <h1>
                <span className={styles.headerIconWrap} aria-hidden>
                  <FaHeartbeat />
                </span>
                Admin Ops / Platform
              </h1>
              <p className={styles.pageSubtitle}>
                Source unique :{' '}
                <code className={styles.inlineCode}>GET /api/v1/platform/status</code>
                <span className={styles.subtle}> — lecture seule</span>
              </p>
            </div>
            <div className={styles.headerActions}>
              <button type="button" className={styles.refreshBtn} onClick={load}>
                Actualiser
              </button>
              <span className={styles.metaLine}>
                Dernière mise à jour : {lastOk ? formatTime(lastOk) : '—'}
              </span>
            </div>
          </header>

          {loading && !data && <div className={styles.loading}>Chargement…</div>}

          {error && (
            <div className={styles.errors} role="alert">
              <strong>Incident de collecte</strong> — {error}
            </div>
          )}

          {data && (
            <>
              <ConfigHint data={data} />

              <div className={styles.summaryStrip}>
                <span className={styles.summaryLabel}>État global</span>
                <StatusBadge status={data.overall_status} />
                <span className={styles.summaryMeta}>
                  généré {formatTime(data.generated_at)}
                </span>
              </div>

              <p className={styles.sectionLabel}>Environnements</p>
              <div className={styles.grid}>
                <EnvCard title="ATMR Production" env={data.environments?.prod} />
                <EnvCard title="ATMR Demo" env={data.environments?.demo} />
              </div>

              <ObservabilityLinks links={data.links} />

              <div className={`${styles.card} ${styles.mutedCard} ${styles.cardSpacedTop}`}>
                <h2 className={styles.cardTitle}>Données techniques</h2>
                <p className={styles.cardMeta}>Version / commit / uptime — non exposés (MVP)</p>
              </div>
            </>
          )}
        </main>
      </div>
    </div>
  );
};

export default AdminPlatformOps;
