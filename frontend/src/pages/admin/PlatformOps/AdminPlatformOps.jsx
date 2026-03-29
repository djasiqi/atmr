import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { FaHeartbeat, FaMicrochip, FaServer } from 'react-icons/fa';
import HeaderDashboard from '../../../components/layout/Header/HeaderDashboard';
import AdminSidebar from '../../../components/layout/Sidebar/AdminSidebar/AdminSidebar';
import StatusBadge from '../../../components/platform/StatusBadge';
import {
  fetchPlatformAuditEvents,
  fetchPlatformAuditReplay,
  fetchPlatformReconciliation,
  fetchPlatformRuntime,
  fetchPlatformStatus,
  fetchPlatformTenant,
  postPlatformPoliciesEvaluate,
  postPlatformRunbookExecution,
  postPlatformRunbookRollback,
  postPlatformSearch,
  postPlatformTenantSuspend,
  postPlatformTenantSuspendPreview,
} from '../../../services/adminService';
import styles from './AdminPlatformOps.module.css';

const CRITICALITY_ORDER = { critical: 0, high: 1, medium: 2, low: 3 };
const CHECK_ORDER = ['ready', 'database', 'redis', 'websocket'];

const REFRESH_OPTIONS = [
  { value: 0, label: 'OFF' },
  { value: 10000, label: '10 s' },
  { value: 30000, label: '30 s' },
  { value: 60000, label: '60 s' },
  { value: 300000, label: '5 min' },
];

function formatTime(iso) {
  if (!iso) return '—';
  try {
    return new Date(iso).toLocaleString('fr-CH');
  } catch {
    return String(iso);
  }
}

function formatRelativeAge(iso) {
  if (!iso) return '—';
  try {
    const d = new Date(iso);
    const sec = Math.round((Date.now() - d.getTime()) / 1000);
    if (sec < 5) return 'à l’instant';
    if (sec < 60) return `il y a ${sec} s`;
    const min = Math.floor(sec / 60);
    if (min < 60) return `il y a ${min} min`;
    const h = Math.floor(min / 60);
    return `il y a ${h} h`;
  } catch {
    return '—';
  }
}

function globalStatus(data) {
  return data?.global_status ?? data?.overall_status ?? 'unknown';
}

function maxLatencyMs(data) {
  let max = null;
  const envs = data?.environments || {};
  for (const env of Object.values(envs)) {
    if (!env?.monitored) continue;
    const checks = env.checks || {};
    for (const c of Object.values(checks)) {
      if (c && typeof c.latency_ms === 'number') {
        const v = c.latency_ms;
        max = max == null ? v : Math.max(max, v);
      }
    }
    if (env.latency_ms != null) {
      const v = env.latency_ms;
      max = max == null ? v : Math.max(max, v);
    }
  }
  return max;
}

/**
 * Incidents dérivés (V1) — plan §8 : critical/high unknown, tout down.
 */
function deriveIncidents(data) {
  const out = [];
  if (!data?.environments) return out;
  for (const [envKey, env] of Object.entries(data.environments)) {
    if (!env.monitored) continue;
    const checks = env.checks || {};
    for (const [name, c] of Object.entries(checks)) {
      if (!c || typeof c !== 'object') continue;
      const st = String(c.status || '').toLowerCase();
      const crit = String(c.criticality || 'medium').toLowerCase();
      if (st === 'down') {
        out.push({
          id: `${envKey}:${name}:down`,
          severity: crit === 'critical' ? 'critical' : 'high',
          status: 'open',
          component: name,
          environment: envKey,
          summary: `${name} (${envKey}) : indisponible`,
          started_at: null,
          recommended_action: null,
        });
      } else if (st === 'unknown' && (crit === 'critical' || crit === 'high')) {
        out.push({
          id: `${envKey}:${name}:unknown`,
          severity: crit === 'critical' ? 'critical' : 'high',
          status: 'open',
          component: name,
          environment: envKey,
          summary: `${name} (${envKey}) : état inconnu`,
          started_at: null,
          recommended_action: null,
        });
      }
    }
  }
  return out;
}

function sortedCheckEntries(checks) {
  const entries = Object.entries(checks || {});
  entries.sort((a, b) => {
    const ca = CRITICALITY_ORDER[a[1]?.criticality] ?? 99;
    const cb = CRITICALITY_ORDER[b[1]?.criticality] ?? 99;
    if (ca !== cb) return ca - cb;
    const ia = CHECK_ORDER.indexOf(a[0]);
    const ib = CHECK_ORDER.indexOf(b[0]);
    if (ia === -1 && ib === -1) return a[0].localeCompare(b[0]);
    if (ia === -1) return 1;
    if (ib === -1) return -1;
    return ia - ib;
  });
  return entries;
}

const RUNTIME_SECTION_ORDER = [
  'process',
  'redis',
  'celery',
  'websocket',
  'dispatch',
  'gps_pipeline',
];

const RUNTIME_SECTION_LABELS = {
  process: 'Processus',
  redis: 'Redis',
  celery: 'Celery',
  websocket: 'WebSocket',
  dispatch: 'Dispatch',
  gps_pipeline: 'Pipeline GPS',
};

const RUNTIME_DATA_KEY_PRIORITY = {
  process: ['pid', 'python_version'],
  redis: [
    'ping_ok',
    'used_memory_human',
    'used_memory_bytes',
    'connected_clients',
    'uptime_in_seconds',
    'evicted_keys',
    'keyspace_hits',
    'keyspace_misses',
    'available',
  ],
  celery: ['inspect_ok', 'workers_count', 'workers', 'broker_transport', 'available'],
  websocket: [],
  dispatch: [],
  gps_pipeline: [],
};

const RUNTIME_FIELD_LABELS = {
  pid: 'PID',
  python_version: 'Python',
  used_memory_human: 'Mémoire',
  used_memory_bytes: 'Mémoire (octets)',
  connected_clients: 'Clients connectés',
  uptime_in_seconds: 'Uptime (s)',
  evicted_keys: 'Clés évincées',
  keyspace_hits: 'Hits clé',
  keyspace_misses: 'Misses clé',
  available: 'Disponible',
  ping_ok: 'Ping OK',
  inspect_ok: 'Inspect OK',
  workers_count: 'Nombre de workers',
  workers: 'Workers',
  broker_transport: 'Broker',
};

function formatRuntimeValue(key, value) {
  if (value === null || value === undefined) return null;
  if (Array.isArray(value)) {
    if (value.length === 0) return null;
    const head = value.slice(0, 5).join(', ');
    return value.length > 5 ? `${head}…` : head;
  }
  if (typeof value === 'boolean') return value ? 'oui' : 'non';
  if (typeof value === 'object') return null;
  return String(value);
}

function buildRuntimeDataRows(sectionKey, data, max = 6) {
  if (!data || typeof data !== 'object') return [];
  const priority = RUNTIME_DATA_KEY_PRIORITY[sectionKey] || [];
  const rows = [];
  const used = new Set();
  for (const k of priority) {
    if (rows.length >= max) break;
    if (!(k in data)) continue;
    const v = formatRuntimeValue(k, data[k]);
    if (v === null) continue;
    used.add(k);
    rows.push({
      key: k,
      label: RUNTIME_FIELD_LABELS[k] || k,
      value: v,
    });
  }
  for (const k of Object.keys(data)) {
    if (rows.length >= max) break;
    if (used.has(k)) continue;
    const v = formatRuntimeValue(k, data[k]);
    if (v === null) continue;
    rows.push({
      key: k,
      label: RUNTIME_FIELD_LABELS[k] || k,
      value: v,
    });
  }
  return rows;
}

function RuntimeSectionBlock({ sectionKey, section }) {
  const title = RUNTIME_SECTION_LABELS[sectionKey] || sectionKey;
  const st = String(section?.status ?? 'unknown').toLowerCase();
  const subtle = st === 'not_implemented';
  const rows = buildRuntimeDataRows(sectionKey, section?.data, 6);
  const showReason = st !== 'ok' && section?.reason;

  return (
    <div className={subtle ? styles.runtimeSectionSubtle : styles.runtimeSection}>
      <div className={styles.runtimeSectionHead}>
        <span className={styles.runtimeSectionTitle}>{title}</span>
        <StatusBadge status={section?.status} />
      </div>
      {showReason && (
        <p
          className={subtle ? styles.runtimeReasonMuted : styles.runtimeReason}
          role="status"
        >
          {section.reason}
        </p>
      )}
      <p className={styles.runtimeChecked}>Mesure : {formatTime(section?.checked_at)}</p>
      {rows.length > 0 && (
        <ul className={styles.runtimeDataList}>
          {rows.map((r) => (
            <li key={r.key}>
              <span className={styles.runtimeDataLabel}>{r.label}</span>
              <span className={styles.runtimeDataValue}>{r.value}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function EnvCard({ env, demoOptional }) {
  if (!env) return null;
  const { monitored, status, latency_ms: latencyMs, checks = {}, errors = [] } = env;
  const isOptionalDemo = Boolean(demoOptional && !monitored);
  const badgeTitle = monitored
    ? 'Statut agrégé'
    : isOptionalDemo
      ? 'Aucune URL démo publique (PLATFORM_API_URL_DEMO) — optionnel si vous n’exposez pas d’API démo'
      : 'Environnement non suivi';
  const showFriendlyDemoCallout =
    isOptionalDemo &&
    errors.some((e) => e.type === 'not_monitored' || /PLATFORM_API_URL_DEMO/i.test(String(e.message || '')));

  const title = env.name || (demoOptional ? 'ATMR Demo' : 'ATMR Production');

  return (
    <div className={styles.card}>
      <h2 className={styles.cardTitle}>
        <FaServer className={styles.cardIcon} aria-hidden />
        {title}
      </h2>
      <div className={styles.cardMeta}>
        <StatusBadge
          status={status}
          title={badgeTitle}
          labelOverride={isOptionalDemo ? 'Non configuré' : undefined}
        />
        {monitored && latencyMs != null && (
          <span>· dernière mesure ~ {Math.round(latencyMs)} ms</span>
        )}
        {!monitored && !isOptionalDemo && <span>· collecte désactivée (config)</span>}
        {isOptionalDemo && (
          <span>· optionnel — pas d’URL démo publique configurée sur l’API</span>
        )}
      </div>
      {monitored && (
        <ul className={styles.checks}>
          {sortedCheckEntries(checks).map(([k, v]) => (
            <li key={k}>
              <strong>{k}</strong> : <StatusBadge status={v?.status} />
              {v?.latency_ms != null && (
                <span className={styles.checkMeta}> · {Math.round(v.latency_ms)} ms</span>
              )}
              {v?.detail && (
                <span className={styles.checkDetail} title={v.detail}>
                  {' '}
                  · {v.detail}
                </span>
              )}
            </li>
          ))}
        </ul>
      )}
      {errors.length > 0 && !showFriendlyDemoCallout && (
        <div className={styles.callout}>
          {errors.map((e, i) => (
            <div key={i}>
              {e.type ? `[${e.type}] ` : ''}
              {e.message}
            </div>
          ))}
        </div>
      )}
      {showFriendlyDemoCallout && (
        <div className={`${styles.callout} ${styles.calloutNeutral}`} role="note">
          Démo non configurée. Comportement attendu si aucune API démo publique n’est exposée. Pour
          activer les checks, définir{' '}
          <code className={styles.inlineCode}>PLATFORM_API_URL_DEMO</code> sur le serveur (URL de base
          sans <code className={styles.inlineCode}>/api/v1</code>), puis redémarrer l’API.
        </div>
      )}
    </div>
  );
}

function ConfigHint({ data }) {
  if (!data) return null;
  const prod = data.environments?.prod;
  const demo = data.environments?.demo;
  const links = data.links || {};
  const needProd = !prod?.monitored;
  const needDemo = !demo?.monitored;
  const needObs = !links.grafana && !links.prometheus && !links.alertmanager;
  if (!needProd && !needDemo && !needObs) return null;
  return (
    <details className={styles.configDetails}>
      <summary className={styles.configSummary}>
        Certains checks peuvent être absents — configuration serveur
      </summary>
      <div className={styles.configHint} role="note">
        Les checks d’environnement s’affichent uniquement si les URLs cibles sont configurées sur le{' '}
        <strong>serveur API</strong>. Définir au minimum <code>PLATFORM_API_URL_PROD</code>, puis
        redémarrer l’API. Voir <code>backend/env.example</code> et{' '}
        <code>backend/docs/PLATFORM_ENV.md</code>.
      </div>
    </details>
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

const AdminPlatformOps = () => {
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

  const [runtime, setRuntime] = useState(null);
  const [runtimeLoading, setRuntimeLoading] = useState(false);
  const [runtimeError, setRuntimeError] = useState(null);

  const [govTenantId, setGovTenantId] = useState('');
  const [govJustification, setGovJustification] = useState('');
  const [govTenantDetail, setGovTenantDetail] = useState(null);
  const [govPreview, setGovPreview] = useState(null);
  const [govSuspendResult, setGovSuspendResult] = useState(null);
  const [govPolicyResult, setGovPolicyResult] = useState(null);
  const [govError, setGovError] = useState(null);
  const [govBusy, setGovBusy] = useState(false);
  const [govSearchQuery, setGovSearchQuery] = useState('');
  const [govSearchResult, setGovSearchResult] = useState(null);
  const [govReconResult, setGovReconResult] = useState(null);
  const [govAuditSample, setGovAuditSample] = useState(null);
  const [govReplayCid, setGovReplayCid] = useState('');
  const [govReplayResult, setGovReplayResult] = useState(null);

  const govTenantIdParsed = useMemo(() => {
    const n = Number.parseInt(String(govTenantId).trim(), 10);
    return Number.isFinite(n) && n > 0 ? n : null;
  }, [govTenantId]);

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

  const loadGovTenant = useCallback(async () => {
    const tid = govTenantIdParsed;
    if (!tid) {
      setGovError('Indiquez un identifiant tenant (entreprise) numérique valide.');
      return;
    }
    setGovError(null);
    setGovBusy(true);
    try {
      const json = await fetchPlatformTenant(tid);
      setGovTenantDetail(json);
    } catch (e) {
      setGovTenantDetail(null);
      setGovError(e?.response?.data?.message || e?.message || 'Chargement tenant impossible');
    } finally {
      setGovBusy(false);
    }
  }, [govTenantIdParsed]);

  const runGovPreview = useCallback(async () => {
    const tid = govTenantIdParsed;
    if (!tid) {
      setGovError('ID tenant invalide.');
      return;
    }
    setGovError(null);
    setGovBusy(true);
    try {
      const json = await postPlatformTenantSuspendPreview(tid, {});
      setGovPreview(json);
    } catch (e) {
      setGovPreview(null);
      setGovError(e?.response?.data?.message || e?.message || 'Preview impossible');
    } finally {
      setGovBusy(false);
    }
  }, [govTenantIdParsed]);

  const runGovPolicyEvaluate = useCallback(async () => {
    const tid = govTenantIdParsed;
    if (!tid) {
      setGovError('ID tenant invalide.');
      return;
    }
    setGovError(null);
    setGovBusy(true);
    try {
      const json = await postPlatformPoliciesEvaluate({
        action_type: 'governance.tenant.suspend',
        scope_type: 'tenant',
        scope_id: String(tid),
      });
      setGovPolicyResult(json);
    } catch (e) {
      setGovPolicyResult(null);
      setGovError(e?.response?.data?.message || e?.message || 'Évaluation policy impossible');
    } finally {
      setGovBusy(false);
    }
  }, [govTenantIdParsed]);

  const runGovSuspend = useCallback(async () => {
    const tid = govTenantIdParsed;
    if (!tid) {
      setGovError('ID tenant invalide.');
      return;
    }
    setGovError(null);
    setGovBusy(true);
    try {
      const json = await postPlatformTenantSuspend(tid, { justification: govJustification });
      setGovSuspendResult(json);
      if (json?.tenant) {
        setGovTenantDetail(json.tenant);
      }
    } catch (e) {
      setGovSuspendResult(e?.response?.data || null);
      setGovError(e?.response?.data?.message || e?.message || 'Suspension impossible');
    } finally {
      setGovBusy(false);
    }
  }, [govTenantIdParsed, govJustification]);

  const runPostSuspendVerify = useCallback(async () => {
    const tid = govTenantIdParsed;
    if (!tid) {
      setGovError('ID tenant invalide.');
      return;
    }
    setGovError(null);
    setGovBusy(true);
    try {
      const json = await postPlatformRunbookExecution('tenant_post_suspend_verify', { tenant_id: tid });
      setGovSuspendResult((prev) => ({ ...(prev || {}), runbook_verify: json }));
    } catch (e) {
      setGovError(e?.response?.data?.message || e?.message || 'Runbook impossible');
    } finally {
      setGovBusy(false);
    }
  }, [govTenantIdParsed]);

  const runGovSearch = useCallback(async () => {
    const q = govSearchQuery.trim();
    if (!q) {
      setGovError('Saisissez une requête (ID tenant, booking, ou UUID utilisateur).');
      return;
    }
    setGovError(null);
    setGovBusy(true);
    try {
      const json = await postPlatformSearch({ query: q });
      setGovSearchResult(json);
    } catch (e) {
      setGovSearchResult(null);
      setGovError(e?.response?.data?.message || e?.message || 'Recherche impossible');
    } finally {
      setGovBusy(false);
    }
  }, [govSearchQuery]);

  const runGovReconciliation = useCallback(async () => {
    const tid = govTenantIdParsed;
    if (!tid) {
      setGovError('ID tenant requis pour la réconciliation.');
      return;
    }
    setGovError(null);
    setGovBusy(true);
    try {
      const json = await fetchPlatformReconciliation(tid);
      setGovReconResult(json);
    } catch (e) {
      setGovReconResult(null);
      setGovError(e?.response?.data?.message || e?.message || 'Réconciliation impossible');
    } finally {
      setGovBusy(false);
    }
  }, [govTenantIdParsed]);

  const loadGovAuditSample = useCallback(async () => {
    const tid = govTenantIdParsed;
    setGovError(null);
    setGovBusy(true);
    try {
      const json = await fetchPlatformAuditEvents({
        per_page: 15,
        page: 1,
        ...(tid ? { company_id: tid } : {}),
        action_category: 'platform_ops',
      });
      setGovAuditSample(json);
    } catch (e) {
      setGovAuditSample(null);
      setGovError(e?.response?.data?.message || e?.message || 'Audit impossible');
    } finally {
      setGovBusy(false);
    }
  }, [govTenantIdParsed]);

  const loadGovReplay = useCallback(async () => {
    const cid = govReplayCid.trim();
    if (!cid) {
      setGovError('Indiquez un correlation_id pour le replay.');
      return;
    }
    setGovError(null);
    setGovBusy(true);
    try {
      const json = await fetchPlatformAuditReplay(cid);
      setGovReplayResult(json);
    } catch (e) {
      setGovReplayResult(null);
      setGovError(e?.response?.data?.message || e?.message || 'Replay impossible');
    } finally {
      setGovBusy(false);
    }
  }, [govReplayCid]);

  const runGovRollbackLastRunbook = useCallback(async () => {
    const exId = govSuspendResult?.runbook_verify?.id;
    if (!exId) {
      setGovError(
        'Aucun execution_id : lancez d’abord « Runbook : vérif post-suspension » (réponse JSON ci-dessous).'
      );
      return;
    }
    setGovError(null);
    setGovBusy(true);
    try {
      const json = await postPlatformRunbookRollback(exId);
      setGovSuspendResult((prev) => ({ ...(prev || {}), runbook_rollback: json }));
    } catch (e) {
      setGovError(e?.response?.data?.message || e?.message || 'Rollback impossible');
    } finally {
      setGovBusy(false);
    }
  }, [govSuspendResult]);

  const loadRuntime = useCallback(async () => {
    setRuntimeError(null);
    setRuntimeLoading(true);
    try {
      const json = await fetchPlatformRuntime();
      setRuntime(json);
    } catch (e) {
      const msg =
        e?.response?.status === 403
          ? 'Accès refusé (403). Vérifiez le rôle admin et la whitelist IP.'
          : e?.response?.data?.message || e?.message || 'Données runtime indisponibles';
      setRuntimeError(msg);
    } finally {
      setRuntimeLoading(false);
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
                Vue de supervision basée sur{' '}
                <code className={styles.inlineCode}>GET /api/v1/platform/status</code>
                <span className={styles.subtle}> — lecture seule</span>
              </p>
            </div>
            <div className={styles.headerActions}>
              <div className={styles.refreshRow}>
                <label htmlFor="platform-poll-interval" className={styles.refreshLabel}>
                  Auto-refresh
                </label>
                <select
                  id="platform-poll-interval"
                  className={styles.refreshSelect}
                  value={pollIntervalMs}
                  onChange={(e) => setPollIntervalMs(Number(e.target.value))}
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
                  Actualiser
                </button>
              </div>
              <span className={styles.metaLine}>
                Dernière mise à jour : {lastOk ? formatTime(lastOk) : '—'}
                {pollIntervalMs > 0 && !pollPaused && !tabHidden && (
                  <span className={styles.subtle}> · intervalle {pollIntervalMs / 1000} s</span>
                )}
                {(pollPaused || pollIntervalMs === 0 || tabHidden) && (
                  <span className={styles.subtle}>
                    {' '}
                    · {pollPaused ? 'pause' : pollIntervalMs === 0 ? 'OFF' : 'onglet inactif'}
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
            <>
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
                <span className={styles.summaryMeta}>
                  généré {formatTime(data.generated_at)}
                </span>
              </div>

              <section
                className={`${styles.card} ${styles.cardSpacedTop} ${styles.govPanel}`}
                aria-labelledby="gov-tenant-heading"
              >
                <h2 id="gov-tenant-heading" className={styles.cardTitle}>
                  Gouvernance tenant (slice V1)
                </h2>
                <p className={styles.cardMeta}>
                  Tenant = entreprise (<code className={styles.inlineCode}>company.id</code>). Affiche{' '}
                  <code className={styles.inlineCode}>desired_state</code>,{' '}
                  <code className={styles.inlineCode}>observed_state</code>,{' '}
                  <code className={styles.inlineCode}>effective_state</code> et{' '}
                  <code className={styles.inlineCode}>reconciliation_status</code> — voir{' '}
                  <code className={styles.inlineCode}>docs/platform/spec-normative-v1.md</code>.
                </p>
                <div className={styles.govRow}>
                  <label className={styles.govLabel} htmlFor="gov-tenant-id">
                    ID tenant
                    <input
                      id="gov-tenant-id"
                      className={styles.govInput}
                      type="number"
                      min={1}
                      value={govTenantId}
                      onChange={(e) => setGovTenantId(e.target.value)}
                      placeholder="ex. 12"
                    />
                  </label>
                  <div className={styles.govActions}>
                    <button
                      type="button"
                      className={styles.govBtn}
                      onClick={loadGovTenant}
                      disabled={govBusy}
                    >
                      Charger l’état
                    </button>
                    <button
                      type="button"
                      className={styles.govBtn}
                      onClick={runGovPolicyEvaluate}
                      disabled={govBusy}
                    >
                      Évaluer policy (suspend)
                    </button>
                    <button
                      type="button"
                      className={styles.govBtn}
                      onClick={runGovPreview}
                      disabled={govBusy}
                    >
                      Prévisualiser impact
                    </button>
                  </div>
                </div>
                <label className={styles.govLabel} htmlFor="gov-justification">
                  Justification (obligatoire pour suspendre)
                  <textarea
                    id="gov-justification"
                    className={styles.govTextarea}
                    value={govJustification}
                    onChange={(e) => setGovJustification(e.target.value)}
                    placeholder="Motif opérationnel (≥ 3 caractères)"
                  />
                </label>
                <div className={styles.govActions}>
                  <button
                    type="button"
                    className={`${styles.govBtn} ${styles.govBtnPrimary}`}
                    onClick={runGovSuspend}
                    disabled={govBusy}
                  >
                    Suspendre le tenant
                  </button>
                  <button
                    type="button"
                    className={styles.govBtn}
                    onClick={runPostSuspendVerify}
                    disabled={govBusy}
                  >
                    Runbook : vérif post-suspension
                  </button>
                  <button
                    type="button"
                    className={styles.govBtn}
                    onClick={runGovReconciliation}
                    disabled={govBusy}
                  >
                    Drift / réconciliation
                  </button>
                  <button
                    type="button"
                    className={styles.govBtn}
                    onClick={loadGovAuditSample}
                    disabled={govBusy}
                  >
                    Échantillon audit (ops)
                  </button>
                  <button
                    type="button"
                    className={styles.govBtn}
                    onClick={runGovRollbackLastRunbook}
                    disabled={govBusy}
                  >
                    Rollback dernière exécution runbook
                  </button>
                </div>
                <p className={styles.sectionLabel}>Replay audit (correlation_id)</p>
                <p className={styles.cardMeta}>
                  Affiche la réponse API telle quelle (pas de recalcul côté navigateur).
                </p>
                <div className={styles.govRow}>
                  <label className={styles.govLabel} htmlFor="gov-replay-cid">
                    correlation_id
                    <input
                      id="gov-replay-cid"
                      className={styles.govInput}
                      value={govReplayCid}
                      onChange={(e) => setGovReplayCid(e.target.value)}
                      placeholder="ex. depuis X-Correlation-Id ou réponse suspend"
                    />
                  </label>
                  <button
                    type="button"
                    className={styles.govBtn}
                    onClick={loadGovReplay}
                    disabled={govBusy}
                  >
                    Charger le replay
                  </button>
                </div>
                <p className={styles.sectionLabel}>Investigation (IDs)</p>
                <div className={styles.govRow}>
                  <label className={styles.govLabel} htmlFor="gov-search-q">
                    Recherche plateforme
                    <input
                      id="gov-search-q"
                      className={styles.govInput}
                      value={govSearchQuery}
                      onChange={(e) => setGovSearchQuery(e.target.value)}
                      placeholder="tenant, booking ou UUID user"
                    />
                  </label>
                  <button
                    type="button"
                    className={styles.govBtn}
                    onClick={runGovSearch}
                    disabled={govBusy}
                  >
                    Rechercher
                  </button>
                </div>
                {govError && (
                  <p className={styles.govError} role="alert">
                    {govError}
                  </p>
                )}
                {govTenantDetail && (
                  <div>
                    <p className={styles.sectionLabel}>État tenant</p>
                    <pre className={styles.govPre}>{JSON.stringify(govTenantDetail, null, 2)}</pre>
                  </div>
                )}
                {govPolicyResult && (
                  <div>
                    <p className={styles.sectionLabel}>Policy evaluate</p>
                    <pre className={styles.govPre}>{JSON.stringify(govPolicyResult, null, 2)}</pre>
                  </div>
                )}
                {govPreview && (
                  <div>
                    <p className={styles.sectionLabel}>Preview blast radius</p>
                    <pre className={styles.govPre}>{JSON.stringify(govPreview, null, 2)}</pre>
                  </div>
                )}
                {govSuspendResult && (
                  <div>
                    <p className={styles.sectionLabel}>Dernière action (suspend / runbook)</p>
                    <pre className={styles.govPre}>{JSON.stringify(govSuspendResult, null, 2)}</pre>
                  </div>
                )}
                {govSearchResult && (
                  <div>
                    <p className={styles.sectionLabel}>Résultat recherche</p>
                    <pre className={styles.govPre}>{JSON.stringify(govSearchResult, null, 2)}</pre>
                  </div>
                )}
                {govReconResult && (
                  <div>
                    <p className={styles.sectionLabel}>Réconciliation / drift</p>
                    <pre className={styles.govPre}>{JSON.stringify(govReconResult, null, 2)}</pre>
                  </div>
                )}
                {govAuditSample && (
                  <div>
                    <p className={styles.sectionLabel}>Audit (échantillon)</p>
                    <pre className={styles.govPre}>{JSON.stringify(govAuditSample, null, 2)}</pre>
                  </div>
                )}
                {govReplayResult && (
                  <div>
                    <p className={styles.sectionLabel}>Replay (API)</p>
                    <pre className={styles.govPre}>{JSON.stringify(govReplayResult, null, 2)}</pre>
                  </div>
                )}
              </section>

              <section className={styles.runtimeCard} aria-labelledby="runtime-heading">
                <div className={styles.runtimeCardHeader}>
                  <div>
                    <h2 id="runtime-heading" className={styles.cardTitle}>
                      <FaMicrochip className={styles.cardIcon} aria-hidden />
                      Runtime
                    </h2>
                    <p className={styles.runtimeCardIntro}>
                      Données d’exploitation enrichies via{' '}
                      <code className={styles.inlineCode}>GET /api/v1/platform/runtime</code> — chargement
                      <strong> manuel uniquement</strong> (aucun impact sur l’auto-refresh du statut
                      ci-dessus).
                    </p>
                  </div>
                  <button
                    type="button"
                    className={styles.runtimeLoadBtn}
                    onClick={loadRuntime}
                    disabled={runtimeLoading}
                  >
                    {runtimeLoading ? 'Chargement…' : runtime ? 'Actualiser le runtime' : 'Charger le runtime'}
                  </button>
                </div>
                {runtimeError && (
                  <div className={styles.runtimeError} role="alert">
                    Runtime indisponible — {runtimeError}
                  </div>
                )}
                {runtime && !runtimeLoading && (
                  <>
                    <p className={styles.runtimeGenerated}>
                      Généré {formatTime(runtime.generated_at)} · âge relatif{' '}
                      {formatRelativeAge(runtime.generated_at)}
                    </p>
                    {RUNTIME_SECTION_ORDER.map((key) => (
                      <RuntimeSectionBlock
                        key={key}
                        sectionKey={key}
                        section={
                          runtime.sections?.[key] ?? {
                            status: 'unknown',
                            reason: null,
                            checked_at: null,
                            data: null,
                          }
                        }
                      />
                    ))}
                  </>
                )}
                {!runtime && !runtimeLoading && !runtimeError && (
                  <p className={styles.cardMeta} role="status">
                    Aucune donnée runtime chargée. Utilisez le bouton pour interroger l’API sans
                    ralentir la vue statut.
                  </p>
                )}
              </section>

              <p className={styles.sectionLabel}>Environnements</p>
              <div className={styles.grid}>
                <EnvCard env={data.environments?.prod} />
                <EnvCard env={data.environments?.demo} demoOptional />
              </div>

              <ObservabilityLinks
                links={data.deep_links?.observability || data.links}
              />

              <div className={`${styles.card} ${styles.mutedCard} ${styles.cardSpacedTop}`}>
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
                        <strong>Uptime process</strong> : {data.metadata.data.process_uptime_seconds}{' '}
                        s
                      </li>
                    )}
                  </ul>
                )}
                {data.metadata?.status === 'not_configured' && (
                  <p className={styles.cardMeta} role="status">
                    Métadonnées non renseignées ({data.metadata.reason || '—'}). Définir{' '}
                    <code className={styles.inlineCode}>PLATFORM_METADATA_GIT_COMMIT</code> et/ou{' '}
                    <code className={styles.inlineCode}>PLATFORM_METADATA_APP_VERSION</code> sur le
                    serveur API si besoin.
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
            </>
          )}
        </main>
      </div>
    </div>
  );
};

export default AdminPlatformOps;
