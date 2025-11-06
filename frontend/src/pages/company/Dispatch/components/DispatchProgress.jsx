import React, { useState, useEffect } from 'react';
import styles from '../modes/Common.module.css';

/**
 * Composant d'indicateur de progression pour le dispatch
 * Affiche une barre de progression et le statut du dispatch en cours
 */
const DispatchProgress = ({
  isActive = false,
  status = 'idle', // 'idle' | 'queued' | 'processing' | 'completed' | 'failed'
  dispatchRunId: _dispatchRunId = null, // Conservé pour compatibilité mais non utilisé actuellement
  startTime = null,
  estimatedDuration = 30000, // Durée estimée en ms (30s par défaut)
  onClose = null,
  assignmentsCount = null,
}) => {
  const [progress, setProgress] = useState(0);
  const [elapsedTime, setElapsedTime] = useState(0);

  // Calculer la progression et le temps écoulé
  useEffect(() => {
    if (
      !isActive ||
      !startTime ||
      status === 'idle' ||
      status === 'completed' ||
      status === 'failed'
    ) {
      if (status === 'completed') {
        setProgress(100);
      } else if (status === 'failed') {
        setProgress(0);
      } else {
        setProgress(0);
      }
      return;
    }

    const updateProgress = () => {
      const now = Date.now();
      const elapsed = now - startTime;
      setElapsedTime(elapsed);

      // Estimation de progression basée sur le temps écoulé
      // Utilise une courbe logarithmique pour ralentir vers la fin
      let estimatedProgress = Math.min(95, (elapsed / estimatedDuration) * 100);

      // Ajuster selon le statut
      if (status === 'queued') {
        estimatedProgress = Math.min(10, estimatedProgress);
      } else if (status === 'processing') {
        // Accélérer légèrement entre 10% et 90%
        estimatedProgress = 10 + estimatedProgress * 0.85;
      }

      setProgress(estimatedProgress);
    };

    // Mise à jour immédiate
    updateProgress();

    // Mettre à jour toutes les 500ms pour fluidité
    const interval = setInterval(updateProgress, 500);

    return () => clearInterval(interval);
  }, [isActive, status, startTime, estimatedDuration]);

  // Format du temps écoulé
  const formatElapsedTime = (ms) => {
    const seconds = Math.floor(ms / 1000);
    if (seconds < 60) {
      return `${seconds}s`;
    }
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}m ${remainingSeconds}s`;
  };

  // Message selon le statut
  const getStatusMessage = () => {
    switch (status) {
      case 'queued':
        return "En file d'attente...";
      case 'processing':
        return 'Optimisation en cours...';
      case 'completed':
        return assignmentsCount
          ? `✅ Dispatch terminé ! ${assignmentsCount} course(s) assignée(s)`
          : '✅ Dispatch terminé !';
      case 'failed':
        return '❌ Erreur lors du dispatch';
      default:
        return 'Prêt';
    }
  };

  // Couleur selon le statut
  const getStatusColor = () => {
    switch (status) {
      case 'queued':
        return 'var(--text-tertiary)';
      case 'processing':
        return 'var(--brand-primary)';
      case 'completed':
        return 'var(--success-primary)';
      case 'failed':
        return 'var(--error-primary)';
      default:
        return 'var(--text-tertiary)';
    }
  };

  if (!isActive && status === 'idle') {
    return null;
  }

  return (
    <div className={styles.dispatchProgressContainer}>
      <div className={styles.dispatchProgressHeader}>
        <div className={styles.dispatchProgressInfo}>
          <span className={styles.dispatchProgressIcon}>
            {status === 'completed' && '✅'}
            {status === 'failed' && '❌'}
            {status === 'processing' && '⏳'}
            {status === 'queued' && '📋'}
          </span>
          <div className={styles.dispatchProgressText}>
            <span className={styles.dispatchProgressTitle}>{getStatusMessage()}</span>
            {status !== 'completed' && status !== 'failed' && elapsedTime > 0 && (
              <span className={styles.dispatchProgressTime}>
                Temps écoulé: {formatElapsedTime(elapsedTime)}
              </span>
            )}
          </div>
        </div>
        {(status === 'completed' || status === 'failed') && onClose && (
          <button className={styles.dispatchProgressClose} onClick={onClose} aria-label="Fermer">
            ×
          </button>
        )}
      </div>

      {(status === 'processing' || status === 'queued') && (
        <div className={styles.dispatchProgressBar}>
          <div
            className={styles.dispatchProgressFill}
            style={{
              width: `${progress}%`,
              backgroundColor: getStatusColor(),
              transition: 'width 0.5s ease-out',
            }}
          />
          <div className={styles.dispatchProgressLabel}>{Math.round(progress)}%</div>
        </div>
      )}

      {status === 'completed' && progress === 100 && (
        <div className={styles.dispatchProgressBar}>
          <div
            className={styles.dispatchProgressFill}
            style={{
              width: '100%',
              backgroundColor: getStatusColor(),
            }}
          />
        </div>
      )}
    </div>
  );
};

export default DispatchProgress;
