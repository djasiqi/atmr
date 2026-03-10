import React from 'react';
import { Link } from 'react-router-dom';
import styles from '../ContactSubpages.module.css';

const SuccessCard = ({ traceId }) => {
  const onCopy = async () => {
    if (!traceId || !navigator?.clipboard?.writeText) {
      return;
    }
    try {
      await navigator.clipboard.writeText(traceId);
    } catch (_) {
      // no-op
    }
  };

  return (
    <section className={styles.successCard} role="status" aria-live="polite">
      <p>Merci. Votre demande a ete transmise.</p>
      <p>
        Reference : <code>{traceId || 'ct_...'}</code>
      </p>
      <p>Nous revenons vers vous sous 24h ouvrees.</p>
      <div className={styles.successActions}>
        <button type="button" className={styles.secondaryButton} onClick={onCopy} disabled={!traceId}>
          Copier la reference
        </button>
        <Link className={styles.primaryLink} to="/contact">
          Revenir aux categories
        </Link>
      </div>
    </section>
  );
};

export default SuccessCard;
