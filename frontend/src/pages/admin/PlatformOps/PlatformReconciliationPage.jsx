import React, { useCallback, useMemo, useState } from 'react';
import { fetchPlatformReconciliation } from '../../../services/adminService';
import styles from './AdminPlatformOps.module.css';
import { GovJsonBlock } from './platformOpsShared';

const PlatformReconciliationPage = () => {
  const [tenantId, setTenantId] = useState('');
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  const tid = useMemo(() => {
    const n = Number.parseInt(String(tenantId).trim(), 10);
    return Number.isFinite(n) && n > 0 ? n : null;
  }, [tenantId]);

  const run = useCallback(async () => {
    if (!tid) {
      setError('Indiquez un identifiant tenant numérique valide.');
      return;
    }
    setError(null);
    setBusy(true);
    try {
      const json = await fetchPlatformReconciliation(tid);
      setResult(json);
    } catch (e) {
      setResult(null);
      setError(e?.response?.data?.message || e?.message || 'Réconciliation impossible');
    } finally {
      setBusy(false);
    }
  }, [tid]);

  return (
    <>
      <header className={styles.pageHeader}>
        <div className={styles.pageHeaderText}>
          <h1>Réconciliation</h1>
          <p className={styles.pageSubtitle}>
            <code className={styles.inlineCode}>GET /api/v1/platform/reconciliation</code> — tenant
            requis
          </p>
        </div>
      </header>
      <div className={styles.tabPanel}>
        <section className={`${styles.card} ${styles.govPanel}`}>
          <div className={styles.govRow}>
            <label className={styles.govLabel} htmlFor="recon-tenant-id">
              ID tenant
              <input
                id="recon-tenant-id"
                className={styles.govInput}
                type="number"
                min={1}
                value={tenantId}
                onChange={(e) => setTenantId(e.target.value)}
                placeholder="ex. 12"
              />
            </label>
            <button type="button" className={styles.govBtn} onClick={run} disabled={busy}>
              {busy ? 'Chargement…' : 'Charger la réconciliation'}
            </button>
          </div>
          {error && (
            <div className={styles.govAlert} role="alert">
              {error}
            </div>
          )}
          <GovJsonBlock title="Réconciliation / drift" data={result} />
        </section>
      </div>
    </>
  );
};

export default PlatformReconciliationPage;
