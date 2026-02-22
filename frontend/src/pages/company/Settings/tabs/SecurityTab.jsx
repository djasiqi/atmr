import React, { useState, useEffect, useCallback } from 'react';
import {
  FiMonitor,
  FiLock,
  FiShield,
  FiActivity,
  FiDownload,
  FiChevronDown,
  FiBell,
  FiKey,
  FiClock,
  FiAlertCircle,
  FiInfo,
} from 'react-icons/fi';
import { formatDistanceToNow, format, isToday, isYesterday } from 'date-fns';
import { fr } from 'date-fns/locale';
import ConfirmationModal from '../../../../components/common/ConfirmationModal';
import {
  fetchSessions,
  revokeSession,
  revokeOtherSessions,
  fetchAuditLogs,
  exportAuditLogs,
  fetchTotpStatus,
  setupTotp,
  verifyTotp,
  disableTotp,
  regenerateRecoveryCodes,
  fetchSecurityPolicy,
  updateSecurityPolicy,
  fetchSecurityAlerts,
  fetchAlertPreferences,
  updateAlertPreferences,
  AUDIT_ACTION_LABELS,
  AUDIT_CATEGORY_LABELS,
} from '../../../../services/securityService';
import styles from '../CompanySettings.module.css';
import s from './SecurityTab.module.css';
import n from './NotificationsTab.module.css';

// ─── Sessions Card ──────────────────────────────────────────

function SessionsCard() {
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [revokeAllOpen, setRevokeAllOpen] = useState(false);
  const [revokeTarget, setRevokeTarget] = useState(null);

  const load = useCallback(async () => {
    try {
      const { data } = await fetchSessions();
      setSessions(data.sessions || []);
    } catch {
      setSessions([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const handleRevoke = async (id) => {
    try {
      await revokeSession(id);
      setSessions((prev) => prev.filter((sess) => sess.id !== id));
    } catch { /* noop */ }
    setRevokeTarget(null);
  };

  const handleRevokeAll = async () => {
    try {
      await revokeOtherSessions();
      setSessions((prev) => prev.filter((sess) => sess.is_current));
    } catch { /* noop */ }
    setRevokeAllOpen(false);
  };

  const otherCount = sessions.filter((sess) => !sess.is_current).length;

  return (
    <div className={`${styles.card} ${s.compactCard}`}>
      <div className={styles.cardHeader}>
        <div className={styles.cardIcon}><FiMonitor size={16} /></div>
        <div className={styles.cardHeaderText}>
          <h3 className={styles.cardTitle}>Sessions actives</h3>
          <p className={styles.cardHint}>
            {loading ? '...' : sessions.length === 0 ? 'Aucune session' : `${sessions.length} appareil(s)`}
          </p>
        </div>
      </div>

      {loading ? (
        <div className={s.emptyState}>Chargement...</div>
      ) : sessions.length === 0 ? (
        <div className={s.emptyState}>Aucune session active.</div>
      ) : (
        <>
          <div className={s.sessionList}>
            {sessions.map((sess) => (
              <div key={sess.id} className={s.sessionRow}>
                <div className={s.sessionInfo}>
                  <span className={s.sessionDevice}>
                    {sess.device_name || 'Appareil inconnu'}
                    {sess.is_current && <span className={s.sessionBadge}>Actuel</span>}
                  </span>
                  <span className={s.sessionMeta}>
                    <span className={s.sessionIp}>{sess.ip_masked}</span>
                    {sess.last_used_at && (
                      <> · {formatDistanceToNow(new Date(sess.last_used_at), { addSuffix: true, locale: fr })}</>
                    )}
                  </span>
                </div>
                {!sess.is_current && (
                  <div className={s.sessionActions}>
                    <button type="button" className={s.revokeBtn} onClick={() => setRevokeTarget(sess)}>
                      Révoquer
                    </button>
                  </div>
                )}
              </div>
            ))}
          </div>

          {otherCount > 0 && (
            <div className={s.sessionFooter}>
              <button type="button" className={s.revokeAllBtn} onClick={() => setRevokeAllOpen(true)}>
                Déconnecter tous les autres appareils
              </button>
            </div>
          )}
        </>
      )}

      <ConfirmationModal
        isOpen={!!revokeTarget}
        onClose={() => setRevokeTarget(null)}
        onConfirm={() => handleRevoke(revokeTarget?.id)}
        title="Déconnecter cet appareil"
        message={`Voulez-vous déconnecter « ${revokeTarget?.device_name || 'cet appareil'} » ?`}
        confirmText="Déconnecter"
        confirmButtonVariant="danger"
      />
      <ConfirmationModal
        isOpen={revokeAllOpen}
        onClose={() => setRevokeAllOpen(false)}
        onConfirm={handleRevokeAll}
        title="Déconnecter tous les autres"
        message="Seul cet appareil restera connecté. Continuer ?"
        confirmText="Tout déconnecter"
        confirmButtonVariant="danger"
      />
    </div>
  );
}

// ─── Audit Log Card ─────────────────────────────────────────

const PERIOD_OPTIONS = [
  { value: '1', label: 'Dernières 24h' },
  { value: '7', label: '7 jours' },
  { value: '30', label: '30 jours' },
];

function getDayLabel(dateStr) {
  const d = new Date(dateStr);
  if (isToday(d)) return "Aujourd'hui";
  if (isYesterday(d)) return 'Hier';
  return format(d, 'd MMMM yyyy', { locale: fr });
}

function formatTime(dateStr) {
  const d = new Date(dateStr);
  if (isToday(d) || isYesterday(d)) return format(d, 'HH:mm');
  return format(d, 'dd/MM HH:mm');
}

function groupByDay(logs) {
  const groups = [];
  let currentLabel = null;
  for (const log of logs) {
    const label = getDayLabel(log.created_at);
    if (label !== currentLabel) {
      groups.push({ label, logs: [] });
      currentLabel = label;
    }
    groups[groups.length - 1].logs.push(log);
  }
  return groups;
}

function AuditLogCard() {
  const [logs, setLogs] = useState([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [hasMore, setHasMore] = useState(false);
  const [category, setCategory] = useState('all');
  const [periodDays, setPeriodDays] = useState('7');
  const [loading, setLoading] = useState(true);
  const [showExport, setShowExport] = useState(false);
  const [exportFormat, setExportFormat] = useState('xlsx');
  const [exporting, setExporting] = useState(false);
  const [catOpen, setCatOpen] = useState(false);

  const load = useCallback(async (pg = 1, append = false) => {
    setLoading(!append);
    try {
      const now = new Date();
      const from = new Date(now - parseInt(periodDays, 10) * 86400000);
      const params = {
        page: pg,
        per_page: 20,
        date_from: from.toISOString(),
        date_to: now.toISOString(),
      };
      if (category !== 'all') params.category = category;

      const { data } = await fetchAuditLogs(params);
      const newLogs = data.logs || [];
      setLogs((prev) => (append ? [...prev, ...newLogs] : newLogs));
      setTotal(data.total || 0);
      setHasMore(data.has_more || false);
      setPage(pg);
    } catch {
      if (!append) setLogs([]);
    } finally {
      setLoading(false);
    }
  }, [category, periodDays]);

  useEffect(() => { load(1); }, [load]);

  const handleExport = async () => {
    setExporting(true);
    try {
      const now = new Date();
      const from = new Date(now - parseInt(periodDays, 10) * 86400000);
      const params = {
        format: exportFormat,
        date_from: from.toISOString(),
        date_to: now.toISOString(),
      };
      if (category !== 'all') params.category = category;

      const { data } = await exportAuditLogs(params);

      const mimeTypes = {
        xlsx: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        csv: 'text/csv;charset=utf-8;',
      };
      const ts = format(now, 'yyyyMMdd_HHmm');
      const blob = new Blob([data], { type: mimeTypes[exportFormat] || mimeTypes.xlsx });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `journal_activite_${ts}.${exportFormat}`;
      a.click();
      URL.revokeObjectURL(url);
    } catch { /* noop */ }
    setExporting(false);
  };

  const grouped = groupByDay(logs);

  return (
    <div className={styles.card}>
      <div className={styles.cardHeader}>
        <div className={styles.cardIcon}><FiActivity size={16} /></div>
        <div className={styles.cardHeaderText}>
          <h3 className={styles.cardTitle}>Journal d&apos;activité</h3>
          <p className={styles.cardHint}>
            {loading ? '...' : `${total} événement(s)`}
          </p>
        </div>
      </div>

      <div className={s.filterBar}>
        <div className={s.filterDropdown}>
          <button
            type="button"
            className={s.filterDropdownBtn}
            onClick={() => setCatOpen((v) => !v)}
          >
            {AUDIT_CATEGORY_LABELS[category] || 'Tous'}
            <FiChevronDown size={12} style={{ transform: catOpen ? 'rotate(180deg)' : 'none', transition: '0.2s' }} />
          </button>
          {catOpen && (
            <div className={s.filterDropdownMenu}>
              {Object.entries(AUDIT_CATEGORY_LABELS).map(([key, label]) => (
                <button
                  key={key}
                  type="button"
                  className={`${s.filterDropdownItem} ${category === key ? s.filterDropdownItemActive : ''}`}
                  onClick={() => { setCategory(key); setCatOpen(false); }}
                >
                  {label}
                </button>
              ))}
            </div>
          )}
        </div>
        <div className={s.filterChips}>
          {PERIOD_OPTIONS.map((opt) => (
            <button
              key={opt.value}
              type="button"
              className={`${s.filterChip} ${periodDays === opt.value ? s.filterChipActive : ''}`}
              onClick={() => setPeriodDays(opt.value)}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      {loading ? (
        <div className={s.emptyState}>Chargement...</div>
      ) : logs.length === 0 ? (
        <div className={s.emptyState}>Aucune activité sur cette période.</div>
      ) : (
        <div className={s.auditList}>
          {grouped.map((group) => (
            <React.Fragment key={group.label}>
              <div className={s.auditDayHeader}>{group.label}</div>
              {group.logs.map((log) => (
                <div key={log.id} className={s.auditRow}>
                  <div className={s.auditMain}>
                    <span className={log.result_status === 'failure' ? s.auditActionFailure : s.auditAction}>
                      {AUDIT_ACTION_LABELS[log.action_type] || log.action_type}
                      {log.result_status === 'failure' && ' — Échec'}
                    </span>
                    <span className={s.auditActor}>
                      {log.actor_username || log.actor_user_type || 'Système'}
                      {log.ip_masked && <> · {log.ip_masked}</>}
                    </span>
                  </div>
                  <span className={s.auditTime}>{formatTime(log.created_at)}</span>
                </div>
              ))}
            </React.Fragment>
          ))}

          {hasMore && (
            <button type="button" className={s.loadMoreBtn} onClick={() => load(page + 1, true)}>
              Afficher plus
            </button>
          )}
        </div>
      )}

      <div className={s.exportPanel}>
        <button
          type="button"
          className={`${styles.button} ${styles.secondary}`}
          onClick={() => setShowExport((v) => !v)}
          style={{ display: 'flex', alignItems: 'center', gap: 5, height: 30, fontSize: 12 }}
        >
          <FiDownload size={12} />
          Exporter
          <FiChevronDown size={11} style={{ transform: showExport ? 'rotate(180deg)' : 'none', transition: '0.2s' }} />
        </button>

        {showExport && (
          <>
            <div className={s.exportRow}>
              <span className={s.exportLabel}>Format</span>
              <div className={s.filterChips}>
                {[{ v: 'xlsx', l: 'Excel' }, { v: 'csv', l: 'CSV' }].map((f) => (
                  <button
                    key={f.v}
                    type="button"
                    className={`${s.filterChip} ${exportFormat === f.v ? s.filterChipActive : ''}`}
                    onClick={() => setExportFormat(f.v)}
                  >
                    {f.l}
                  </button>
                ))}
              </div>
            </div>
            <p className={s.exportHint}>
              Les adresses IP sont masquées dans l&apos;export.
            </p>
            <div className={s.exportActions}>
              <button
                type="button"
                className={`${styles.button} ${styles.secondary}`}
                onClick={handleExport}
                disabled={exporting}
                style={{ height: 30, fontSize: 12 }}
              >
                {exporting ? 'Export...' : 'Télécharger'}
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

// ─── Two Factor Card ────────────────────────────────────────

function TwoFactorCard() {
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [step, setStep] = useState(null);
  const [setupData, setSetupData] = useState(null);
  const [code, setCode] = useState('');
  const [password, setPassword] = useState('');
  const [recoveryCodes, setRecoveryCodes] = useState([]);
  const [error, setError] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const loadStatus = useCallback(async () => {
    try {
      const { data } = await fetchTotpStatus();
      setStatus(data);
    } catch {
      setStatus({ enabled: false });
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { loadStatus(); }, [loadStatus]);

  const handleSetup = async () => {
    setError('');
    setSubmitting(true);
    try {
      const { data } = await setupTotp();
      setSetupData(data);
      setStep('qr');
    } catch (e) {
      setError(e.response?.data?.error || 'Erreur lors de la configuration');
    }
    setSubmitting(false);
  };

  const handleVerify = async () => {
    setError('');
    setSubmitting(true);
    try {
      const { data } = await verifyTotp(code);
      setRecoveryCodes(data.recovery_codes || []);
      setStep('codes');
      setCode('');
    } catch (e) {
      setError(e.response?.data?.error || 'Code invalide');
    }
    setSubmitting(false);
  };

  const handleDisable = async () => {
    setError('');
    setSubmitting(true);
    try {
      await disableTotp(password);
      setStep(null);
      setPassword('');
      loadStatus();
    } catch (e) {
      setError(e.response?.data?.error || 'Mot de passe incorrect');
    }
    setSubmitting(false);
  };

  const handleRegenerate = async () => {
    setError('');
    setSubmitting(true);
    try {
      const { data } = await regenerateRecoveryCodes(code);
      setRecoveryCodes(data.recovery_codes || []);
      setStep('codes');
      setCode('');
      loadStatus();
    } catch (e) {
      setError(e.response?.data?.error || 'Code invalide');
    }
    setSubmitting(false);
  };

  const finishSetup = () => {
    setStep(null);
    setSetupData(null);
    setRecoveryCodes([]);
    loadStatus();
  };

  if (loading) {
    return (
      <div className={`${styles.card} ${s.compactCard}`}>
        <div className={styles.cardHeader}>
          <div className={styles.cardIcon}><FiLock size={16} /></div>
          <div className={styles.cardHeaderText}>
            <h3 className={styles.cardTitle}>Validation en deux étapes</h3>
            <p className={styles.cardHint}>Chargement...</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={`${styles.card} ${s.compactCard}`}>
      <div className={styles.cardHeader}>
        <div className={styles.cardIcon}><FiLock size={16} /></div>
        <div className={styles.cardHeaderText}>
          <h3 className={styles.cardTitle}>Validation en deux étapes</h3>
          <p className={styles.cardHint}>
            {status?.enabled ? 'Activée' : 'Non configurée'}
          </p>
        </div>
      </div>

      <div className={s.tfaBody}>
        {error && <p className={s.tfaError}>{error}</p>}

        {/* Not enabled — default */}
        {!status?.enabled && !step && (
          <div className={s.tfaDefaultRow}>
            <p className={s.tfaText} style={{ margin: 0 }}>
              Protégez votre compte avec une vérification supplémentaire.
            </p>
            <button
              type="button"
              className={`${styles.button} ${styles.secondary}`}
              onClick={handleSetup}
              disabled={submitting}
              style={{ height: 30, fontSize: 11, whiteSpace: 'nowrap' }}
            >
              {submitting ? '...' : 'Activer'}
            </button>
          </div>
        )}

        {/* QR code step */}
        {step === 'qr' && setupData && (
          <>
            <p className={s.tfaText}>
              Scannez ce QR code avec Google Authenticator, Authy ou similaire.
            </p>
            {setupData.qr_code_base64 && (
              <img src={setupData.qr_code_base64} alt="QR Code TOTP" className={s.tfaQr} />
            )}
            <p className={s.tfaSecret}>Clé : {setupData.secret_display}</p>
            <input
              type="text"
              inputMode="numeric"
              maxLength={6}
              placeholder="000 000"
              value={code}
              onChange={(e) => setCode(e.target.value.replace(/\D/g, '').slice(0, 6))}
              className={s.tfaCodeInput}
            />
            <button
              type="button"
              className={`${styles.button} ${styles.secondary}`}
              onClick={handleVerify}
              disabled={submitting || code.length !== 6}
              style={{ height: 30, fontSize: 12, width: '100%' }}
            >
              {submitting ? 'Vérification...' : 'Vérifier et activer'}
            </button>
          </>
        )}

        {/* Recovery codes */}
        {step === 'codes' && (
          <>
            <p className={s.tfaTextBold}>2FA activée avec succès.</p>
            <p className={s.tfaTextDanger}>
              Conservez ces codes. Ils ne seront plus affichés.
            </p>
            <div className={s.tfaCodesGrid}>
              {recoveryCodes.map((c, i) => (
                <div key={i}>{c}</div>
              ))}
            </div>
            <button
              type="button"
              className={`${styles.button} ${styles.secondary}`}
              onClick={finishSetup}
              style={{ height: 30, fontSize: 12, width: '100%' }}
            >
              J&apos;ai sauvegardé mes codes
            </button>
          </>
        )}

        {/* Enabled state */}
        {status?.enabled && !step && (
          <>
            <p className={s.tfaStatus}>
              Activée{status.enabled_at ? ` depuis le ${format(new Date(status.enabled_at), 'd MMM yyyy', { locale: fr })}` : ''}
            </p>
            <p className={s.tfaStatusSub}>
              {status.recovery_codes_remaining} codes de secours restants sur 10
            </p>
            <div className={s.tfaBtnRow}>
              <button
                type="button"
                className={`${styles.button} ${styles.secondary}`}
                onClick={() => { setStep('regenerate'); setCode(''); setError(''); }}
                style={{ height: 28, fontSize: 11 }}
              >
                Régénérer les codes
              </button>
              <button
                type="button"
                className={s.revokeBtn}
                onClick={() => { setStep('disable'); setPassword(''); setError(''); }}
              >
                Désactiver
              </button>
            </div>
          </>
        )}

        {/* Disable step */}
        {step === 'disable' && (
          <>
            <p className={s.tfaText}>Entrez votre mot de passe pour désactiver.</p>
            <input
              type="password"
              placeholder="Mot de passe"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className={s.tfaPasswordInput}
            />
            <div className={s.tfaBtnRow}>
              <button type="button" className={s.revokeBtn} onClick={handleDisable} disabled={submitting || !password}>
                {submitting ? '...' : 'Désactiver'}
              </button>
              <button
                type="button"
                className={`${styles.button} ${styles.secondary}`}
                onClick={() => setStep(null)}
                style={{ height: 28, fontSize: 11 }}
              >
                Annuler
              </button>
            </div>
          </>
        )}

        {/* Regenerate step */}
        {step === 'regenerate' && (
          <>
            <p className={s.tfaText}>Entrez un code TOTP pour régénérer vos codes de secours.</p>
            <input
              type="text"
              inputMode="numeric"
              maxLength={6}
              placeholder="000 000"
              value={code}
              onChange={(e) => setCode(e.target.value.replace(/\D/g, '').slice(0, 6))}
              className={s.tfaCodeInput}
            />
            <div className={s.tfaBtnRow}>
              <button
                type="button"
                className={`${styles.button} ${styles.secondary}`}
                onClick={handleRegenerate}
                disabled={submitting || code.length !== 6}
                style={{ height: 28, fontSize: 11 }}
              >
                {submitting ? '...' : 'Régénérer'}
              </button>
              <button
                type="button"
                className={`${styles.button} ${styles.secondary}`}
                onClick={() => setStep(null)}
                style={{ height: 28, fontSize: 11 }}
              >
                Annuler
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

// ─── Security Policy Card ───────────────────────────────────

const POLICY_2FA_ROLES = [
  { role: 'admin', label: 'Administrateurs', hint: 'Accès complet' },
  { role: 'manager', label: 'Managers', hint: 'Gestion courante' },
];

const ENFORCEMENT_OPTIONS = [
  { value: 'warn', label: 'Avertissement' },
  { value: 'enforce', label: 'Bloquant' },
];

const EXPIRY_OPTIONS = [
  { value: '', label: 'Jamais' },
  { value: '90', label: '90 jours' },
  { value: '180', label: '180 jours' },
];

const SESSION_OPTIONS = [
  { value: '7', label: '7 jours' },
  { value: '14', label: '14 jours' },
  { value: '30', label: '30 jours' },
  { value: '90', label: '90 jours' },
];

function SecurityPolicyCard() {
  const [policy, setPolicy] = useState(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [openDrop, setOpenDrop] = useState(null);

  useEffect(() => {
    (async () => {
      try {
        const { data } = await fetchSecurityPolicy();
        setPolicy(data.policy);
      } catch {
        setPolicy({
          require_2fa_roles: [],
          password_expiry_days: null,
          max_session_days: 30,
          enforcement_mode: 'warn',
        });
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const save = useCallback(async (updated) => {
    setSaving(true);
    try {
      const { data } = await updateSecurityPolicy(updated);
      setPolicy(data.policy);
    } catch { /* noop */ }
    setSaving(false);
  }, []);

  const toggle2faRole = (role) => {
    if (!policy) return;
    const roles = policy.require_2fa_roles || [];
    const updated = roles.includes(role)
      ? roles.filter((r) => r !== role)
      : [...roles, role];
    const newPolicy = { ...policy, require_2fa_roles: updated };
    setPolicy(newPolicy);
    save(newPolicy);
  };

  const updateField = (field, value) => {
    if (!policy) return;
    const newPolicy = { ...policy, [field]: value };
    setPolicy(newPolicy);
    save(newPolicy);
  };

  if (loading) return null;

  const roles2fa = policy?.require_2fa_roles || [];

  return (
    <div className={styles.card}>
      <div className={styles.cardHeader}>
        <div className={styles.cardIcon}><FiShield size={16} /></div>
        <div className={styles.cardHeaderText}>
          <h3 className={styles.cardTitle}>Politique de sécurité</h3>
          <p className={styles.cardHint}>
            {saving
              ? <span className={s.policySavingBadge}>Enregistrement…</span>
              : 'Règles entreprise'}
          </p>
        </div>
      </div>

      {/* 2FA enforcement — notifRow toggle pattern */}
      <div className={n.notifList} style={{ paddingBottom: 4 }}>
        {POLICY_2FA_ROLES.map((r) => (
          <label key={r.role} className={n.notifRow} htmlFor={`policy-2fa-${r.role}`}>
            <div className={n.notifInfo}>
              <span className={n.notifLabel}>
                <FiKey size={11} style={{ marginRight: 5, verticalAlign: -1, opacity: 0.5 }} />
                2FA {r.label}
              </span>
              <span className={n.notifHint}>{r.hint}</span>
            </div>
            <div className={n.miniToggle}>
              <input
                id={`policy-2fa-${r.role}`}
                type="checkbox"
                checked={roles2fa.includes(r.role)}
                onChange={() => toggle2faRole(r.role)}
              />
              <span className={n.miniSlider} />
            </div>
          </label>
        ))}
      </div>

      {/* Policy settings — dropdown chips */}
      <div className={s.policyBody}>
        {[
          {
            key: 'enforcement',
            icon: <FiAlertCircle size={11} className={s.policyRowIcon} />,
            label: 'Mode',
            options: ENFORCEMENT_OPTIONS,
            current: policy?.enforcement_mode || 'warn',
            onChange: (v) => updateField('enforcement_mode', v),
          },
          {
            key: 'expiry',
            icon: <FiLock size={11} className={s.policyRowIcon} />,
            label: 'Mots de passe',
            options: EXPIRY_OPTIONS,
            current: String(policy?.password_expiry_days || ''),
            onChange: (v) => updateField('password_expiry_days', v ? parseInt(v, 10) : null),
          },
          {
            key: 'session',
            icon: <FiClock size={11} className={s.policyRowIcon} />,
            label: 'Session',
            options: SESSION_OPTIONS,
            current: String(policy?.max_session_days || 30),
            onChange: (v) => updateField('max_session_days', parseInt(v, 10)),
          },
        ].map((cfg) => (
          <div key={cfg.key} className={s.policySection}>
            <div className={s.policyInlineRow}>
              <span className={s.policyRowLabel}>
                {cfg.icon}
                {cfg.label}
              </span>
              <div className={s.filterDropdown}>
                <button
                  type="button"
                  className={s.filterDropdownBtn}
                  onClick={() => setOpenDrop(openDrop === cfg.key ? null : cfg.key)}
                >
                  {cfg.options.find((o) => o.value === cfg.current)?.label || '—'}
                  <FiChevronDown size={12} style={{ transform: openDrop === cfg.key ? 'rotate(180deg)' : 'none', transition: '0.2s' }} />
                </button>
                {openDrop === cfg.key && (
                  <div className={s.filterDropdownMenu}>
                    {cfg.options.map((opt) => (
                      <button
                        key={opt.value}
                        type="button"
                        className={`${s.filterDropdownItem} ${cfg.current === opt.value ? s.filterDropdownItemActive : ''}`}
                        onClick={() => { cfg.onChange(opt.value); setOpenDrop(null); }}
                      >
                        {opt.label}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className={s.policyFooter}>
        <FiInfo size={11} className={s.policyFooterIcon} />
        <p className={s.policyFooterText}>
          S&apos;applique à tous les utilisateurs de l&apos;entreprise.
        </p>
      </div>
    </div>
  );
}

// ─── Security Alerts Card ───────────────────────────────

const ALERT_PREFS = [
  { key: 'failed_logins_burst', label: 'Tentatives échouées', hint: '3+ échecs en 1 heure' },
  { key: 'new_device_login', label: 'Nouvel appareil', hint: 'Appareil inconnu' },
  { key: 'new_country_login', label: 'Nouveau pays', hint: 'Pays inhabituel' },
];

function SecurityAlertsCard() {
  const [alerts, setAlerts] = useState([]);
  const [totalFailed, setTotalFailed] = useState(0);
  const [prefs, setPrefs] = useState({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    (async () => {
      try {
        const [alertsRes, prefsRes] = await Promise.all([
          fetchSecurityAlerts(),
          fetchAlertPreferences(),
        ]);
        setAlerts(alertsRes.data.alerts || []);
        setTotalFailed(alertsRes.data.total_failed_logins_30d || 0);
        setPrefs(prefsRes.data.preferences || {});
      } catch { /* noop */ }
      setLoading(false);
    })();
  }, []);

  const togglePref = async (key) => {
    const updated = { ...prefs, [key]: !prefs[key] };
    setPrefs(updated);
    try {
      await updateAlertPreferences(updated);
    } catch { /* noop */ }
  };

  const activePrefs = ALERT_PREFS.filter((p) => prefs[p.key]).length;

  return (
    <div className={styles.card}>
      <div className={styles.cardHeader}>
        <div className={styles.cardIcon}><FiBell size={16} /></div>
        <div className={styles.cardHeaderText}>
          <h3 className={styles.cardTitle}>Alertes email</h3>
          <p className={styles.cardHint}>
            {loading ? '...' : `${activePrefs} sur ${ALERT_PREFS.length} actives`}
          </p>
        </div>
      </div>

      {loading ? (
        <div className={s.emptyState}>Chargement...</div>
      ) : (
        <>
          {/* Recent alerts */}
          {alerts.length > 0 && (
            <div className={s.auditList} style={{ paddingBottom: 0 }}>
              {alerts.slice(0, 3).map((alert) => (
                <div key={alert.id} className={s.auditRow}>
                  <div className={s.auditMain}>
                    <span className={s.auditActionFailure}>{alert.message}</span>
                    <span className={s.auditActor}>
                      {alert.ip_masked} · {alert.device}
                    </span>
                  </div>
                  <span className={s.auditTime}>
                    {alert.created_at && formatDistanceToNow(new Date(alert.created_at), { addSuffix: true, locale: fr })}
                  </span>
                </div>
              ))}
            </div>
          )}

          {alerts.length === 0 && !totalFailed && (
            <div className={s.emptyState}>Aucune alerte récente.</div>
          )}

          {totalFailed > 0 && (
            <div className={s.policyFooter} style={{ borderTop: alerts.length > 0 ? undefined : 'none' }}>
              <FiAlertCircle size={11} className={s.policyFooterIcon} />
              <p className={s.policyFooterText}>
                {totalFailed} tentative(s) échouée(s) · 30 derniers jours
              </p>
            </div>
          )}

          {/* Prefs — notifRow toggle pattern */}
          <div className={n.notifList}>
            {ALERT_PREFS.map((p) => (
              <label key={p.key} className={n.notifRow} htmlFor={`alert-${p.key}`}>
                <div className={n.notifInfo}>
                  <span className={n.notifLabel}>{p.label}</span>
                  <span className={n.notifHint}>{p.hint}</span>
                </div>
                <div className={n.miniToggle}>
                  <input
                    id={`alert-${p.key}`}
                    type="checkbox"
                    checked={!!prefs[p.key]}
                    onChange={() => togglePref(p.key)}
                  />
                  <span className={n.miniSlider} />
                </div>
              </label>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

// ─── Main SecurityTab ───────────────────────────────────────

const SecurityTab = () => {
  return (
    <div className={`${styles.settingsForm} ${styles.billingFormBlock}`}>
      <div className={s.sectionDivider}>
        <h2>Mon compte</h2>
        <div className={s.sectionLine} />
      </div>
      <div className={styles.billingGrid}>
        <div className={styles.billingCol}>
          <SessionsCard />
        </div>
        <div className={styles.billingCol}>
          <TwoFactorCard />
        </div>
      </div>

      <div className={s.sectionDivider}>
        <h2>Entreprise</h2>
        <div className={s.sectionLine} />
      </div>
      <div className={styles.billingGrid}>
        <div className={styles.billingCol}>
          <AuditLogCard />
        </div>
        <div className={styles.billingCol}>
          <SecurityPolicyCard />
          <SecurityAlertsCard />
        </div>
      </div>
    </div>
  );
};

export default SecurityTab;
