import React, { useCallback, useState } from 'react';
import { FaMicrochip } from 'react-icons/fa';
import { fetchPlatformRuntime } from '../../../services/adminService';
import styles from './AdminPlatformOps.module.css';
import {
  formatRelativeAge,
  formatTime,
  RUNTIME_SECTION_ORDER,
  RuntimeSectionBlock,
} from './platformOpsShared';

const PlatformRuntimePage = () => {
  const [runtime, setRuntime] = useState(null);
  const [runtimeLoading, setRuntimeLoading] = useState(false);
  const [runtimeError, setRuntimeError] = useState(null);

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

  return (
    <>
      <header className={styles.pageHeader}>
        <div className={styles.pageHeaderText}>
          <h1>Runtime</h1>
          <p className={styles.pageSubtitle}>
            <code className={styles.inlineCode}>GET /api/v1/platform/runtime</code> — chargement manuel
          </p>
        </div>
      </header>
      <div className={styles.tabPanel}>
        <section className={styles.runtimeCard} aria-labelledby="runtime-heading">
          <div className={styles.runtimeCardHeader}>
            <div>
              <h2 id="runtime-heading" className={styles.cardTitle}>
                <FaMicrochip className={styles.cardIcon} aria-hidden />
                Runtime
              </h2>
              <p className={styles.runtimeCardIntro}>
                Aucun appel automatique au montage — utilisez le bouton ci-dessous.
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
              Aucune donnée runtime chargée.
            </p>
          )}
        </section>
      </div>
    </>
  );
};

export default PlatformRuntimePage;
