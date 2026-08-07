import React from 'react';
import styles from './SessionIdleWarningToast.module.css';

const WarningIcon = () => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 24 24"
    fill="currentColor"
    width="18"
    height="18"
    aria-hidden="true"
  >
    <path
      fillRule="evenodd"
      d="M9.401 3.003c1.155-2 4.043-2 5.197 0l7.355 12.748c1.154 2-.29 4.5-2.599 4.5H4.645c-2.309 0-3.752-2.5-2.598-4.5L9.4 3.003zM12 8.25a.75.75 0 01.75.75v3.75a.75.75 0 01-1.5 0V9a.75.75 0 01.75-.75zm0 8.25a.75.75 0 100-1.5.75.75 0 000 1.5z"
      clipRule="evenodd"
    />
  </svg>
);

/**
 * Toast custom — préavis d'inactivité session.
 * Affiché via toast.custom (Sonner).
 */
const SessionIdleWarningToast = ({
  secondsLeft,
  totalSeconds = 60,
  renewing = false,
  onStay,
  onLogout,
}) => {
  const safeTotal = Math.max(1, Number(totalSeconds) || 60);
  const safeLeft = Math.max(0, Math.min(safeTotal, Number(secondsLeft) || 0));
  const progressPct = (safeLeft / safeTotal) * 100;

  return (
    <div
      className={styles.card}
      role="alertdialog"
      aria-labelledby="session-idle-title"
      aria-describedby="session-idle-desc"
    >
      <div className={styles.head}>
        <div className={styles.iconWrap}>
          <WarningIcon />
        </div>
        <div className={styles.copy}>
          <p id="session-idle-title" className={styles.title}>
            Session inactive
          </p>
          <p id="session-idle-desc" className={styles.desc}>
            {renewing
              ? 'Renouvellement de votre session…'
              : 'Vous allez être déconnecté faute d’activité.'}
          </p>
        </div>
        <div className={styles.countdown} aria-live="polite" aria-atomic="true">
          <span className={styles.seconds}>{safeLeft}</span>
          <span className={styles.secondsUnit}>sec</span>
        </div>
      </div>

      <div
        className={styles.progressTrack}
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={safeTotal}
        aria-valuenow={safeLeft}
        aria-label="Temps restant avant déconnexion"
      >
        <div className={styles.progressBar} style={{ width: `${progressPct}%` }} />
      </div>

      <div className={styles.actions}>
        <button
          type="button"
          className={`${styles.btn} ${styles.btnGhost}`}
          disabled={renewing}
          onClick={onLogout}
        >
          Se déconnecter
        </button>
        <button
          type="button"
          className={`${styles.btn} ${styles.btnPrimary}`}
          disabled={renewing}
          onClick={onStay}
        >
          {renewing ? 'Renouvellement…' : 'Rester connecté'}
        </button>
      </div>
    </div>
  );
};

export default SessionIdleWarningToast;

export { styles as sessionIdleWarningToastStyles };
