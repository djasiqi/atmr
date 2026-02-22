import React from 'react';
import PropTypes from 'prop-types';
import { FiInfo } from 'react-icons/fi';

const ProTip = ({ title, message, styles = {} }) => {
  return (
    <div className={styles.proTip}>
      <h4>
        <FiInfo size={14} />
        {title || 'Conseil'}
      </h4>
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
