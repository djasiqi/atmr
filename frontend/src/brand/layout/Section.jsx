import React from 'react';
import styles from './Section.module.css';

/**
 * Conteneur de section de page publique.
 * Neutre : ne connaît pas le « marketing », seulement la mise en page.
 */
export function Section({
  as: Comp = 'section',
  children,
  className = '',
  tone = 'default',
  ...rest
}) {
  const toneClass = tone === 'tint' ? styles.toneTint : tone === 'plain' ? styles.tonePlain : styles.toneDefault;
  return (
    <Comp className={`${styles.section} ${toneClass} ${className}`.trim()} {...rest}>
      {children}
    </Comp>
  );
}

export function SectionHeader({ children, className = '', align = 'start', ...rest }) {
  const alignClass = align === 'center' ? styles.headerCenter : styles.headerStart;
  return (
    <header className={`${styles.header} ${alignClass} ${className}`.trim()} {...rest}>
      {children}
    </header>
  );
}

export function SectionBody({ children, className = '', ...rest }) {
  return (
    <div className={`${styles.body} ${className}`.trim()} {...rest}>
      {children}
    </div>
  );
}

export function SectionFooter({ children, className = '', ...rest }) {
  return (
    <footer className={`${styles.footer} ${className}`.trim()} {...rest}>
      {children}
    </footer>
  );
}

export function SectionEyebrow({ children, className = '', ...rest }) {
  return (
    <div className={`${styles.eyebrow} ${className}`.trim()} {...rest}>
      <span className={styles.eyebrowLine} aria-hidden />
      {children}
    </div>
  );
}

export function SectionTitle({ as: Comp = 'h2', children, className = '', ...rest }) {
  return (
    <Comp className={`${styles.title} ${className}`.trim()} {...rest}>
      {children}
    </Comp>
  );
}

export function SectionLead({ children, className = '', ...rest }) {
  return (
    <p className={`${styles.lead} ${className}`.trim()} {...rest}>
      {children}
    </p>
  );
}
