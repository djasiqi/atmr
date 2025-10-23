import React from 'react';
import PropTypes from 'prop-types';

/**
 * Composant réutilisable pour afficher des conseils professionnels
 */
const ProTip = ({ title = '💡 Conseil Pro', message, styles = {} }) => {
  return (
    <div className={styles.proTip}>
      <h4>{title}</h4>
      <p>{message}</p>
    </div>
  );
};

ProTip.propTypes = {
  title: PropTypes.string,
  message: PropTypes.oneOfType([PropTypes.string, PropTypes.node]).isRequired,
  styles: PropTypes.object,
};

export default ProTip;
