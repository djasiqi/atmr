import React, { useCallback, useState } from 'react';
import { fetchPlatformAuditReplay, postPlatformSearch } from '../../../services/adminService';
import styles from './AdminPlatformOps.module.css';
import { GovJsonBlock } from './platformOpsShared';

const PlatformInvestigationPage = () => {
  const [govSearchQuery, setGovSearchQuery] = useState('');
  const [govSearchResult, setGovSearchResult] = useState(null);
  const [govReplayCid, setGovReplayCid] = useState('');
  const [govReplayResult, setGovReplayResult] = useState(null);
  const [govError, setGovError] = useState(null);
  const [govBusy, setGovBusy] = useState(false);

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

  return (
    <>
      <header className={styles.pageHeader}>
        <div className={styles.pageHeaderText}>
          <h1>Investigation</h1>
          <p className={styles.pageSubtitle}>
            Recherche et replay — aucun appel au montage jusqu’à action utilisateur.
          </p>
        </div>
      </header>
      <div className={styles.tabPanel}>
        <section className={`${styles.card} ${styles.govPanel}`}>
          {govError && (
            <div className={styles.govAlert} role="alert">
              {govError}
            </div>
          )}
          <div className={styles.govGroup}>
            <h3 className={styles.govGroupTitle}>Replay</h3>
            <p className={styles.cardMeta}>Réponse API brute (pas de recalcul navigateur).</p>
            <div className={styles.govRow}>
              <label className={styles.govLabel} htmlFor="inv-replay-cid">
                correlation_id
                <input
                  id="inv-replay-cid"
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
          </div>
          <div className={styles.govGroup}>
            <h3 className={styles.govGroupTitle}>Recherche (IDs)</h3>
            <div className={styles.govRow}>
              <label className={styles.govLabel} htmlFor="inv-search-q">
                Requête
                <input
                  id="inv-search-q"
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
          </div>
          <div className={styles.govGroup}>
            <h3 className={styles.govGroupTitle}>Réponses API</h3>
            <GovJsonBlock title="Résultat recherche" data={govSearchResult} />
            <GovJsonBlock title="Replay (API)" data={govReplayResult} />
          </div>
        </section>
      </div>
    </>
  );
};

export default PlatformInvestigationPage;
