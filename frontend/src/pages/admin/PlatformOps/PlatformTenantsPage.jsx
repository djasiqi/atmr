import React, { useCallback, useMemo, useState } from 'react';
import {
  fetchPlatformTenant,
  postPlatformPoliciesEvaluate,
  postPlatformRunbookExecution,
  postPlatformRunbookRollback,
  postPlatformTenantSuspend,
  postPlatformTenantSuspendPreview,
} from '../../../services/adminService';
import styles from './AdminPlatformOps.module.css';
import { GovJsonBlock } from './platformOpsShared';

const PlatformTenantsPage = () => {
  const [govTenantId, setGovTenantId] = useState('');
  const [govJustification, setGovJustification] = useState('');
  const [govTenantDetail, setGovTenantDetail] = useState(null);
  const [govPreview, setGovPreview] = useState(null);
  const [govSuspendResult, setGovSuspendResult] = useState(null);
  const [govPolicyResult, setGovPolicyResult] = useState(null);
  const [govError, setGovError] = useState(null);
  const [govBusy, setGovBusy] = useState(false);

  const govTenantIdParsed = useMemo(() => {
    const n = Number.parseInt(String(govTenantId).trim(), 10);
    return Number.isFinite(n) && n > 0 ? n : null;
  }, [govTenantId]);

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
      const json = await postPlatformRunbookExecution('tenant_post_suspend_verify', {
        tenant_id: tid,
      });
      setGovSuspendResult((prev) => ({ ...(prev || {}), runbook_verify: json }));
    } catch (e) {
      setGovError(e?.response?.data?.message || e?.message || 'Runbook impossible');
    } finally {
      setGovBusy(false);
    }
  }, [govTenantIdParsed]);

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

  return (
    <>
      <header className={styles.pageHeader}>
        <div className={styles.pageHeaderText}>
          <h1>Tenants — Gouvernance</h1>
          <p className={styles.pageSubtitle}>
            <code className={styles.inlineCode}>company.id</code> · spec{' '}
            <code className={styles.inlineCode}>docs/platform/spec-normative-v1.md</code>
          </p>
        </div>
      </header>

      <div className={styles.tabPanel}>
        <section className={`${styles.card} ${styles.govPanel}`} aria-labelledby="gov-tenant-heading">
          <p className={styles.tabPanelHint}>
            Saisir un ID puis charger — aucune liste implicite au montage.
          </p>
          <h2 id="gov-tenant-heading" className={styles.cardTitle}>
            Gouvernance tenant
          </h2>
          {govError && (
            <div className={styles.govAlert} role="alert">
              {govError}
            </div>
          )}
          <div className={styles.govGroup}>
            <h3 className={styles.govGroupTitle}>Contexte &amp; lecture</h3>
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
          </div>
          <div className={styles.govGroup}>
            <h3 className={styles.govGroupTitle}>Suspension</h3>
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
            </div>
            <div className={styles.govActions}>
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
                onClick={runGovRollbackLastRunbook}
                disabled={govBusy}
              >
                Rollback dernière exécution runbook
              </button>
            </div>
          </div>
          <div className={styles.govGroup}>
            <h3 className={styles.govGroupTitle}>Réponses API</h3>
            <GovJsonBlock title="État tenant" data={govTenantDetail} />
            <GovJsonBlock title="Policy evaluate" data={govPolicyResult} />
            <GovJsonBlock title="Preview blast radius" data={govPreview} />
            <GovJsonBlock title="Dernière action (suspend / runbook)" data={govSuspendResult} />
          </div>
        </section>
      </div>
    </>
  );
};

export default PlatformTenantsPage;
