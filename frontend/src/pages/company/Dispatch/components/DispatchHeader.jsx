import React from 'react';
import {
  FiSettings,
  FiZap,
  FiPlay,
  FiLoader,
  FiSliders,
  FiCheck,
} from 'react-icons/fi';
import InlineDatePicker from '../../../../components/ui/InlineDatePicker';

const MODE_SUBTITLE = {
  manual: 'Mode manuel - aucune assignation automatique',
  semi_auto: 'Mode semi-automatique - suggestions a valider',
  fully_auto: 'Mode automatique - assignation geree par le systeme',
};

const MODE_BADGE = {
  manual: 'Manuel',
  semi_auto: 'Semi-Auto',
  fully_auto: 'Automatique',
};

const DispatchHeader = ({
  date,
  setDate,
  regularFirst,
  setRegularFirst,
  allowEmergency,
  setAllowEmergency,
  onRunDispatch,
  loading,
  dispatchSuccess,
  dispatchProgress = 0,
  dispatchLabel = '',
  dispatchMode = 'semi_auto',
  modeLoading = false,
  styles = {},
  onShowAdvancedSettings,
  hasOverrides = false,
  fastMode = false,
  setFastMode,
}) => {
  const isManual = dispatchMode === 'manual';
  const isFullyAuto = dispatchMode === 'fully_auto';
  const modeLabel =
    modeLoading && dispatchMode == null
      ? null
      : (MODE_BADGE[dispatchMode] || '—');

  return (
    <div className={styles.headerSection}>
      <div className={styles.pageHeader}>
        <div className={styles.titleRow}>
          <div className={styles.titleGroup}>
            <h1 className={styles.dispatchTitle}>Dispatch</h1>
            <span
              className={styles.modeBadge}
              data-tour-id="dispatch-mode-badge"
              data-mode-loading={modeLoading && dispatchMode == null ? 'true' : undefined}
              title={modeLoading && dispatchMode == null ? 'Chargement du mode' : modeLabel}
            >
              {modeLoading && dispatchMode == null ? (
                <FiLoader
                  className={styles.spinIcon}
                  size={15}
                  aria-label="Chargement du mode de dispatch"
                />
              ) : (
                modeLabel
              )}
            </span>
          </div>
          <div className={styles.headerActions}>
            <div data-tour-id="dispatch-date-picker">
              <InlineDatePicker
                value={date}
                onChange={(v) => setDate(v)}
                placeholder="Date"
              />
            </div>
            {isManual && (
              <button
                onClick={() => {
                  const companyId = window.location.pathname.split('/')[3] || '';
                  window.location.href = `/dashboard/company/${companyId}/settings#operations`;
                }}
                className={styles.btnSecondary}
              >
                <FiSettings size={14} />
                Automatisation
              </button>
            )}
          </div>
        </div>
        <p className={styles.modeSubtitle}>
          {modeLoading && dispatchMode == null
            ? 'Chargement du mode de dispatch...'
            : (MODE_SUBTITLE[dispatchMode] || MODE_SUBTITLE.manual)}
        </p>
      </div>

      {dispatchSuccess && <div className={styles.successMessage}>{dispatchSuccess}</div>}

      {loading && dispatchProgress > 0 && (
        <div className={styles.progressBar}>
          <div className={styles.progressFill} style={{ width: `${dispatchProgress}%` }}>
            <span className={styles.progressLabel}>{dispatchLabel || 'En cours...'}</span>
          </div>
          <span className={styles.progressPercent}>{dispatchProgress}%</span>
        </div>
      )}

      {/* Controles semi-auto / fully-auto uniquement */}
      {!isManual && (
        <div className={styles.compactFilters}>
          {isFullyAuto ? null : (
            <>
              <label className={styles.checkboxLabel}>
                <input
                  type="checkbox"
                  checked={regularFirst}
                  onChange={(e) => setRegularFirst(e.target.checked)}
                  className={styles.checkbox}
                />
                Chauffeurs reguliers prioritaires
              </label>
              <label className={styles.checkboxLabel}>
                <input
                  type="checkbox"
                  checked={allowEmergency}
                  onChange={(e) => setAllowEmergency(e.target.checked)}
                  className={styles.checkbox}
                />
                Autoriser chauffeurs d'urgence
              </label>
              {setFastMode && (
                <label
                  className={styles.checkboxLabel}
                  title="Garantit une solution en moins de 1 minute"
                >
                  <input
                    type="checkbox"
                    checked={fastMode}
                    onChange={(e) => setFastMode(e.target.checked)}
                    className={styles.checkbox}
                  />
                  <FiZap size={12} />
                  Dispatch rapide (&lt;1min)
                </label>
              )}
              <button onClick={onRunDispatch} disabled={loading} className={styles.dispatchBtn}>
                {loading ? (
                  <>
                    <FiLoader size={14} className={styles.spinIcon} />
                    En cours...
                  </>
                ) : (
                  <>
                    <FiPlay size={14} />
                    Lancer Dispatch
                  </>
                )}
              </button>

              {onShowAdvancedSettings && (
                <button
                  onClick={onShowAdvancedSettings}
                  className={`${styles.advancedBtn} ${hasOverrides ? styles.hasOverrides : ''}`}
                  title={
                    hasOverrides
                      ? 'Parametres personnalises actifs'
                      : 'Configurer parametres avances'
                  }
                >
                  <FiSliders size={14} />
                  {hasOverrides ? (
                    <>
                      Parametres <FiCheck size={12} />
                    </>
                  ) : (
                    'Avance'
                  )}
                </button>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
};

export default DispatchHeader;
