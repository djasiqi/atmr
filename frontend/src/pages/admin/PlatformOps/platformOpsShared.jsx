import React from 'react';
import { FaMicrochip, FaServer } from 'react-icons/fa';
import StatusBadge from '../../../components/platform/StatusBadge';
import styles from './AdminPlatformOps.module.css';

export const CRITICALITY_ORDER = { critical: 0, high: 1, medium: 2, low: 3 };
export const CHECK_ORDER = ['ready', 'database', 'redis', 'websocket'];

export const REFRESH_OPTIONS = [
  { value: 0, label: 'OFF' },
  { value: 10000, label: '10 s' },
  { value: 30000, label: '30 s' },
  { value: 60000, label: '60 s' },
  { value: 300000, label: '5 min' },
];

export function GovJsonBlock({ title, data }) {
  if (!data) return null;
  return (
    <details className={styles.jsonDetails}>
      <summary className={styles.jsonSummary}>{title}</summary>
      <pre className={styles.govPre}>{JSON.stringify(data, null, 2)}</pre>
    </details>
  );
}

export function formatTime(iso) {
  if (!iso) return '—';
  try {
    return new Date(iso).toLocaleString('fr-CH');
  } catch {
    return String(iso);
  }
}

export function formatRelativeAge(iso) {
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

export function globalStatus(data) {
  return data?.global_status ?? data?.overall_status ?? 'unknown';
}

export function maxLatencyMs(data) {
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

export function deriveIncidents(data) {
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

export function sortedCheckEntries(checks) {
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

export const RUNTIME_SECTION_ORDER = [
  'process',
  'redis',
  'celery',
  'websocket',
  'dispatch',
  'gps_pipeline',
];

export const RUNTIME_SECTION_LABELS = {
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

export function RuntimeSectionBlock({ sectionKey, section }) {
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

export function EnvCard({ env, demoOptional }) {
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

export function ConfigHint({ data }) {
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

export function ObservabilityLinks({ links }) {
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

export { FaMicrochip };
