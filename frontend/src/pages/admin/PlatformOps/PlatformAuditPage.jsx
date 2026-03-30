import React, { useEffect, useState } from 'react';
import { fetchPlatformAuditEvents } from '../../../services/adminService';
import styles from './AdminPlatformOps.module.css';
import { GovJsonBlock } from './platformOpsShared';

const PlatformAuditPage = () => {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      setError(null);
      try {
        const json = await fetchPlatformAuditEvents({
          per_page: 20,
          page: 1,
          action_category: 'platform_ops',
        });
        if (!cancelled) setData(json);
      } catch (e) {
        if (!cancelled) {
          setError(e?.response?.data?.message || e?.message || 'Audit impossible');
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <>
      <header className={styles.pageHeader}>
        <div className={styles.pageHeaderText}>
          <h1>Audit</h1>
          <p className={styles.pageSubtitle}>
            Liste paginée — <code className={styles.inlineCode}>GET /api/v1/platform/audit-events</code>
          </p>
        </div>
      </header>
      <div className={styles.tabPanel}>
        {loading && <div className={styles.loading}>Chargement…</div>}
        {error && (
          <div className={styles.errors} role="alert">
            {error}
          </div>
        )}
        {!loading && !error && <GovJsonBlock title="Événements (page 1)" data={data} />}
      </div>
    </>
  );
};

export default PlatformAuditPage;
