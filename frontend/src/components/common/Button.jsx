// src/components/common/Button.jsx
import React from 'react';
import PropTypes from 'prop-types';
import styles from './Button.module.css';

/**
 * Composant Button centralisé pour toute l'application
 *
 * @param {string} variant - Variante du bouton: 'primary', 'secondary', 'success', 'danger', 'warning', 'ghost'
 * @param {string} size - Taille du bouton: 'sm', 'md', 'lg'
 * @param {boolean} fullWidth - Si true, le bouton prend toute la largeur disponible
 * @param {boolean} loading - Si true, affiche un état de chargement
 * @param {React.ReactNode} children - Contenu du bouton
 * @param {string} className - Classes CSS supplémentaires
 * @param {string} type - Type HTML du bouton: 'button', 'submit', 'reset'
 */
const Button = ({
  variant = 'primary',
  size = 'md',
  fullWidth = false,
  loading = false,
  disabled = false,
  children,
  className = '',
  type = 'button',
  onClick,
  ...props
}) => {
  const classes = [
    styles.button,
    styles[variant],
    styles[size],
    fullWidth && styles.fullWidth,
    loading && styles.loading,
    className,
  ]
    .filter(Boolean)
    .join(' ');

  return (
    <button
      type={type}
      className={classes}
      onClick={onClick}
      disabled={disabled || loading}
      {...props}
    >
      {loading && <span className={styles.spinner}></span>}
      <span className={loading ? styles.contentWithLoader : ''}>{children}</span>
    </button>
  );
};

Button.propTypes = {
  variant: PropTypes.oneOf(['primary', 'secondary', 'success', 'danger', 'warning', 'ghost']),
  size: PropTypes.oneOf(['sm', 'md', 'lg']),
  fullWidth: PropTypes.bool,
  loading: PropTypes.bool,
  disabled: PropTypes.bool,
  children: PropTypes.node.isRequired,
  className: PropTypes.string,
  type: PropTypes.oneOf(['button', 'submit', 'reset']),
  onClick: PropTypes.func,
};

export default Button;
