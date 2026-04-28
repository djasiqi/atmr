import React from 'react';
import styles from './DriverMobilePreferredBanner.module.css';

const storeUrl = process.env.REACT_APP_DRIVER_APP_STORE_URL;

/**
 * Rappel produit : chauffeur terrain canon = app mobile (ADR 012 + LIRIE_MOBILE_WEB_CANON).
 */
export default function DriverMobilePreferredBanner() {
  return (
    <section
      className={styles.banner}
      role="status"
      aria-live="polite"
      title="Chauffeur terrain : l’application mobile est la surface canonique (ADR 012). Le web est complément / secours."
    >
      <div className={styles.title}>Application mobile LIRIE (recommandée)</div>
      <p className={styles.text}>
        Les missions, le suivi GPS, les notifications et le mode hors-ligne sont conçus pour
        l&apos;application mobile. Cette interface web reste un complément ou un secours.
      </p>
      {storeUrl ? (
        <a className={styles.link} href={storeUrl} target="_blank" rel="noopener noreferrer">
          Ouvrir la fiche app / magasin
        </a>
      ) : null}
    </section>
  );
}
