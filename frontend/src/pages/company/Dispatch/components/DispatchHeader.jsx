import React from 'react';

/**
 * Composant d'en-tête pour la page de dispatch
 */
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
  styles = {},
  onShowAdvancedSettings, // 🆕
  hasOverrides = false, // 🆕
  fastMode = false, // ⚡ Mode rapide
  setFastMode, // ⚡ Setter pour mode rapide
}) => {
  const _makeToday = () => {
    const d = new Date();
    const pad = (n) => String(n).padStart(2, '0');
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
  };

  return (
    <div className={styles.headerSection}>
      <div className={styles.pageHeader}>
        <div className={styles.titleRow}>
          <h1>
            {dispatchMode === 'manual'
              ? '✋ Dispatch Manuel'
              : dispatchMode === 'semi_auto'
                ? '⚙️ Dispatch Semi-Automatique'
                : '🤖 Dispatch Automatique'}
          </h1>
          <span className={styles.modeBadge}>
            Mode actuel:{' '}
            {dispatchMode === 'manual'
              ? '✋ Manuel'
              : dispatchMode === 'semi_auto'
                ? '⚙️ Semi-Automatique'
                : '🤖 Totalement Automatique'}
          </span>
        </div>
      </div>

      {dispatchSuccess && <div className={styles.successMessage}>{dispatchSuccess}</div>}

      {/* Barre de progression du dispatch */}
      {loading && dispatchProgress > 0 && (
        <div className={styles.progressBar}>
          <div className={styles.progressFill} style={{ width: `${dispatchProgress}%` }}>
            <span className={styles.progressLabel}>{dispatchLabel || 'En cours...'}</span>
          </div>
          <span className={styles.progressPercent}>{dispatchProgress}%</span>
        </div>
      )}

      <div className={styles.compactFilters}>
        {/* En mode fully_auto, afficher uniquement le sélecteur de date */}
        {dispatchMode === 'fully_auto' ? (
          <input
            type="date"
            value={date}
            onChange={(e) => setDate(e.target.value)}
            className={styles.dateInput}
          />
        ) : (
          <>
            <input
              type="date"
              value={date}
              onChange={(e) => setDate(e.target.value)}
              className={styles.dateInput}
            />
            {/* ⚡ En mode manuel, afficher uniquement le calendrier */}
            {dispatchMode !== 'manual' && (
              <>
                <label className={styles.checkboxLabel}>
                  <input
                    type="checkbox"
                    checked={regularFirst}
                    onChange={(e) => setRegularFirst(e.target.checked)}
                    className={styles.checkbox}
                  />
                  Chauffeurs réguliers prioritaires
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
                {/* ⚡ Option Dispatch rapide */}
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
                    ⚡ Dispatch rapide (&lt;1min)
                  </label>
                )}
                <button onClick={onRunDispatch} disabled={loading} className={styles.dispatchBtn}>
                  {loading ? '⏳ En cours...' : '🚀 Lancer Dispatch'}
                </button>

                {/* 🆕 Bouton paramètres avancés */}
                {onShowAdvancedSettings && (
                  <button
                    onClick={onShowAdvancedSettings}
                    className={`${styles.advancedBtn} ${hasOverrides ? styles.hasOverrides : ''}`}
                    title={
                      hasOverrides
                        ? 'Paramètres personnalisés actifs'
                        : 'Configurer paramètres avancés'
                    }
                  >
                    ⚙️ {hasOverrides ? 'Paramètres ✓' : 'Avancé'}
                  </button>
                )}
              </>
            )}

            <span className={styles.courseCount}>
              {/* On pourrait afficher le nombre de courses ici si disponible */}
            </span>
          </>
        )}
      </div>
    </div>
  );
};

export default DispatchHeader;
