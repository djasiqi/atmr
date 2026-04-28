import React, { useEffect, useState } from 'react';
import styles from './PwaOfflineBanner.module.css';

/**
 * Bandeau global lorsque le navigateur signale l’absence de réseau (complète la page offline.html du SW).
 */
const PwaOfflineBanner = () => {
  const [offline, setOffline] = useState(
    typeof navigator !== 'undefined' ? !navigator.onLine : false
  );

  useEffect(() => {
    const onOff = () => setOffline(!navigator.onLine);
    window.addEventListener('online', onOff);
    window.addEventListener('offline', onOff);
    return () => {
      window.removeEventListener('online', onOff);
      window.removeEventListener('offline', onOff);
    };
  }, []);

  if (!offline) return null;

  return (
    <div className={styles.banner} role="alert" aria-live="assertive">
      <span className={styles.text}>
        Hors ligne : les données affichées peuvent être incomplètes. Reconnectez-vous pour réserver ou payer.
      </span>
    </div>
  );
};

export default PwaOfflineBanner;
