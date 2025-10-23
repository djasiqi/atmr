import React from 'react';
import PropTypes from 'prop-types';
import styles from './EmptyState.module.css';

/**
 * Composant réutilisable pour afficher un état vide
 */
const EmptyState = ({ icon = '📦', title, message, action }) => {
  return (
    <div className={styles.emptyState}>
      <div className={styles.emptyIcon}>{icon}</div>
      <h3 className={styles.emptyTitle}>{title}</h3>
      {message && <p className={styles.emptyMessage}>{message}</p>}
      {action && <div className={styles.emptyAction}>{action}</div>}
    </div>
  );
};

EmptyState.propTypes = {
  icon: PropTypes.string,
  title: PropTypes.string.isRequired,
  message: PropTypes.string,
  action: PropTypes.node,
};

export default EmptyState;
