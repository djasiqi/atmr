import React from 'react';
import PropTypes from 'prop-types';

/**
 * Composant réutilisable pour les bannières d'information de mode
 */
const ModeBanner = ({ icon, title, description, action, variant = 'info', styles = {} }) => {
  const bannerClass = `${styles.dispatchModeBanner || ''} ${styles[variant] || ''}`;

  return (
    <div className={bannerClass}>
      <div className={styles.bannerIcon}>{icon}</div>
      <div className={styles.bannerContent}>
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
      {action && <div className={styles.bannerAction}>{action}</div>}
    </div>
  );
};

ModeBanner.propTypes = {
  icon: PropTypes.string.isRequired,
  title: PropTypes.string.isRequired,
  description: PropTypes.string.isRequired,
  action: PropTypes.node,
  variant: PropTypes.oneOf(['info', 'manual', 'semiAuto', 'fullyAuto']),
  styles: PropTypes.object,
};

export default ModeBanner;
