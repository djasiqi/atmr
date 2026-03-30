import React, { useEffect, useState } from 'react';
import { fetchPlatformRunbooks } from '../../../services/adminService';
import styles from './AdminPlatformOps.module.css';
import { GovJsonBlock } from './platformOpsShared';

const PlatformRunbooksPage = () => {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      setError(null);
      try {
        const json = await fetchPlatformRunbooks();
        if (!cancelled) setData(json);
      } catch (e) {
        if (!cancelled) {
          setError(e?.response?.data?.message || e?.message || 'Chargement impossible');
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
          <h1>Runbooks</h1>
          <p className={styles.pageSubtitle}>
            Catalogue <code className={styles.inlineCode}>GET /api/v1/platform/runbooks</code>
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
        {!loading && !error && <GovJsonBlock title="Catalogue" data={data} />}
      </div>
    </>
  );
};

export default PlatformRunbooksPage;
