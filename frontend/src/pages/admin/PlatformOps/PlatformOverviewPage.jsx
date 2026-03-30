import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { FaHeartbeat } from 'react-icons/fa';
import { fetchPlatformStatus } from '../../../services/adminService';
import StatusBadge from '../../../components/platform/StatusBadge';
import styles from './AdminPlatformOps.module.css';
import {
  ConfigHint,
  deriveIncidents,
  formatRelativeAge,
  formatTime,
  globalStatus,
  maxLatencyMs,
  ObservabilityLinks,
  REFRESH_OPTIONS,
  EnvCard,
} from './platformOpsShared';

const PlatformOverviewPage = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastOk, setLastOk] = useState(null);
  const [pollIntervalMs, setPollIntervalMs] = useState(30000);
  const [pollPaused, setPollPaused] = useState(false);
  const [visibilityTick, setVisibilityTick] = useState(0);
  const [tabHidden, setTabHidden] = useState(false);
  const [lastStateChangeAt, setLastStateChangeAt] = useState(null);
  const prevGlobalRef = useRef(null);

  const load = useCallback(async () => {
    setError(null);
    try {
      const json = await fetchPlatformStatus();
      setData(json);
      const nowIso = new Date().toISOString();
      setLastOk(nowIso);
      const g = globalStatus(json);
      if (prevGlobalRef.current !== null && prevGlobalRef.current !== g) {
        setLastStateChangeAt(nowIso);
      }
      prevGlobalRef.current = g;
    } catch (e) {
      const msg =
        e?.response?.status === 403
          ? 'Accès refusé (403). Vérifiez le rôle admin et la whitelist IP.'
          : e?.response?.status === 502
            ? 'Passerelle 502 : l’API est injoignable ou redémarre (souvent variable SOCKETIO_CORS_ORIGINS manquante en prod — voir logs backend).'
            : e?.response?.data?.message || e?.message || 'Erreur de chargement';
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const syncHidden = () => {
      if (typeof document === 'undefined') return;
      setTabHidden(document.hidden);
    };
    syncHidden();
    const onVis = () => {
      syncHidden();
      setVisibilityTick((t) => t + 1);
      if (typeof document !== 'undefined' && !document.hidden) {
        load();
      }
    };
    document.addEventListener('visibilitychange', onVis);
    return () => document.removeEventListener('visibilitychange', onVis);
  }, [load]);

  useEffect(() => {
    load();
  }, [load]);

  useEffect(() => {
    if (pollPaused || pollIntervalMs === 0 || tabHidden) return undefined;
    const id = setInterval(load, pollIntervalMs);
    return () => clearInterval(id);
  }, [load, pollIntervalMs, pollPaused, visibilityTick, tabHidden]);

  const incidents = useMemo(() => (data ? deriveIncidents(data) : []), [data]);
  const summary = data?.summary;
  const gs = data ? globalStatus(data) : null;
  const worstLat = data ? maxLatencyMs(data) : null;

  const monitoredEnvCount = data
    ? [data.environments?.prod, data.environments?.demo].filter((e) => e?.monitored).length
    : 0;

  return (
    <>
      <header className={styles.pageHeader}>
        <div className={styles.pageHeaderText}>
          <h1>
            <span className={styles.headerIconWrap} aria-hidden>
              <FaHeartbeat />
            </span>
            Plateforme — Vue globale
          </h1>
          <p className={styles.pageSubtitle}>
            Santé des environnements et liens observabilité —{' '}
            <code className={styles.inlineCode}>GET /api/v1/platform/status</code>
          </p>
        </div>
        <div className={styles.headerActions}>
          <div className={styles.toolbar}>
            <label htmlFor="platform-poll-interval" className={styles.refreshLabel}>
              Rafraîchir
            </label>
            <select
              id="platform-poll-interval"
              className={styles.refreshSelect}
              value={pollIntervalMs}
              onChange={(e) => setPollIntervalMs(Number(e.target.value))}
              aria-label="Intervalle de rafraîchissement automatique du statut"
            >
              {REFRESH_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))}
            </select>
            <button
              type="button"
              className={styles.pauseBtn}
              onClick={() => setPollPaused((p) => !p)}
              aria-pressed={pollPaused}
            >
              {pollPaused ? 'Reprendre' : 'Pause'}
            </button>
            <button type="button" className={styles.refreshBtn} onClick={load}>
              Actualiser le statut
            </button>
          </div>
          <span className={styles.metaLine}>
            Statut : {lastOk ? formatTime(lastOk) : '—'}
            {pollIntervalMs > 0 && !pollPaused && !tabHidden && (
              <span className={styles.subtle}> · toutes les {pollIntervalMs / 1000} s</span>
            )}
            {(pollPaused || pollIntervalMs === 0 || tabHidden) && (
              <span className={styles.subtle}>
                {' '}
                · {pollPaused ? 'pause' : pollIntervalMs === 0 ? 'auto OFF' : 'onglet en arrière-plan'}
              </span>
            )}
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
        <div className={styles.tabPanel} role="tabpanel" id="platform-panel-overview">
          <p className={styles.tabPanelHint}>
            Source principale :{' '}
            <code className={styles.inlineCode}>GET /api/v1/platform/status</code> (agrégat lecture
            seule, rafraîchi selon la barre ci-dessus).
          </p>
          <ConfigHint data={data} />

          {incidents.length > 0 && (
            <section className={styles.incidentBanner} aria-label="Incidents en cours">
              <h2 className={styles.incidentTitle}>Incidents en cours</h2>
              <ul className={styles.incidentList}>
                {incidents.map((inc) => (
                  <li key={inc.id}>{inc.summary}</li>
                ))}
              </ul>
            </section>
          )}

          {incidents.length === 0 && (
            <p className={styles.noIncident} role="status">
              Aucun incident actif (vue instantanée dérivée du statut).
            </p>
          )}

          <div className={styles.summaryStrip}>
            <span className={styles.summaryLabel}>État global</span>
            <StatusBadge status={gs} />
            <span className={styles.summaryMetaInline}>
              {monitoredEnvCount} env. surveillé{monitoredEnvCount > 1 ? 's' : ''}
            </span>
            {summary && (
              <span className={styles.summaryMetaInline}>
                Checks : {summary.ok_checks ?? 0}/{summary.total_checks ?? 0} OK
                {summary.degraded_checks > 0 && ` · ${summary.degraded_checks} dégradés`}
                {summary.down_checks > 0 && ` · ${summary.down_checks} hors service`}
                {summary.unknown_checks > 0 && ` · ${summary.unknown_checks} inconnus`}
              </span>
            )}
            {worstLat != null && (
              <span className={styles.summaryMetaInline}>Latence max : {Math.round(worstLat)} ms</span>
            )}
            <span className={styles.summaryMetaInline}>
              Données : {formatRelativeAge(data.generated_at || lastOk)}
            </span>
            {lastStateChangeAt && (
              <span className={styles.summaryMetaInline}>
                Dernier changement d’état : {formatTime(lastStateChangeAt)}
              </span>
            )}
            <span className={styles.summaryMeta}>généré {formatTime(data.generated_at)}</span>
          </div>

          <p className={styles.sectionLabel}>Environnements</p>
          <div className={styles.grid}>
            <EnvCard env={data.environments?.prod} />
            <EnvCard env={data.environments?.demo} demoOptional />
          </div>

          <div className={styles.overviewFoot}>
            <ObservabilityLinks links={data.deep_links?.observability || data.links} />
            <div className={`${styles.card} ${styles.mutedCard}`}>
              <h2 className={styles.cardTitle}>Données techniques</h2>
              {data.metadata?.status === 'ok' && data.metadata?.data && (
                <ul className={styles.techList}>
                  {data.metadata.data.app_version && (
                    <li>
                      <strong>Version applicative</strong> : {data.metadata.data.app_version}
                    </li>
                  )}
                  {data.metadata.data.git_commit && (
                    <li>
                      <strong>Commit Git</strong> :{' '}
                      <code className={styles.inlineCode}>{data.metadata.data.git_commit}</code>
                    </li>
                  )}
                  {data.metadata.data.process_uptime_seconds != null && (
                    <li>
                      <strong>Uptime process</strong> : {data.metadata.data.process_uptime_seconds} s
                    </li>
                  )}
                </ul>
              )}
              {data.metadata?.status === 'not_configured' && (
                <p className={styles.cardMeta} role="status">
                  Métadonnées non renseignées ({data.metadata.reason || '—'}). Définir{' '}
                  <code className={styles.inlineCode}>PLATFORM_METADATA_GIT_COMMIT</code> et/ou{' '}
                  <code className={styles.inlineCode}>PLATFORM_METADATA_APP_VERSION</code> sur le serveur
                  API si besoin.
                </p>
              )}
              {data.metadata?.status === 'not_implemented' && (
                <ul className={styles.techList}>
                  <li>
                    <strong>Image Docker</strong> : non exposée
                  </li>
                  <li>
                    <strong>Commit Git</strong> : non exposé
                  </li>
                  <li>
                    <strong>Uptime process</strong> : non exposé
                  </li>
                </ul>
              )}
              <p className={styles.cardMeta}>
                Source : champ <code className={styles.inlineCode}>metadata</code> de{' '}
                <code className={styles.inlineCode}>GET /api/v1/platform/status</code>.
              </p>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default PlatformOverviewPage;
