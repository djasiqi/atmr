import React, { useEffect, useMemo, useState } from 'react';
import styles from './DataFreshnessBadge.module.css';

function formatAge(ms) {
  if (!Number.isFinite(ms) || ms < 0) return 'à l’instant';
  const sec = Math.floor(ms / 1000);
  if (sec < 60) return `il y a ${sec}s`;
  const min = Math.floor(sec / 60);
  if (min < 60) return `il y a ${min} min`;
  const hr = Math.floor(min / 60);
  return `il y a ${hr} h`;
}

/**
 * Métadonnées d’affichage (niveau + texte) pour le badge ou une zone d’état unique.
 */
export function computeDataFreshnessMeta({
  isOnline = true,
  isSyncing = false,
  lastSyncAt = null,
  realtimeEnabled = false,
  realtimeConnected = true,
  realtimeDegraded = false,
  sourceLabel = 'Données',
  now = Date.now(),
}) {
  if (!isOnline) {
    return {
      level: 'offline',
      text: 'Hors ligne : affichage des dernières données disponibles.',
    };
  }
  if (isSyncing) {
    return {
      level: 'recent',
      text: 'Mise à jour des données en cours…',
    };
  }
  if (realtimeDegraded) {
    return {
      level: 'partial',
      text: 'Temps réel dégradé : séquence indisponible. Rechargement manuel recommandé.',
    };
  }
  if (!lastSyncAt) {
    return {
      level: 'stale',
      text: `${sourceLabel} : première lecture en cours.`,
    };
  }

  const ageMs = Math.max(0, now - lastSyncAt);
  const ageLabel = formatAge(ageMs);
  const isRealtimeUnavailable = realtimeEnabled && realtimeConnected === false;

  if (isRealtimeUnavailable) {
    return {
      level: 'partial',
      text: `Suivi en direct momentanément limité (${ageLabel}). Les statuts peuvent être affichés avec un léger retard ; la réservation et cette page fonctionnent normalement.`,
    };
  }
  if (ageMs < 30_000) {
    return {
      level: 'fresh',
      text: `Dernière mise à jour ${ageLabel}.`,
    };
  }
  if (ageMs < 120_000) {
    return {
      level: 'recent',
      text: `Données récentes, mises à jour ${ageLabel}.`,
    };
  }
  return {
    level: 'stale',
    text: `Données potentiellement retardées (mise à jour ${ageLabel}).`,
  };
}

export default function DataFreshnessBadge({
  lastSyncAt = null,
  isSyncing = false,
  realtimeEnabled = false,
  realtimeConnected = true,
  realtimeDegraded = false,
  sourceLabel = 'Données',
  className = '',
}) {
  const [now, setNow] = useState(Date.now());
  const [isOnline, setIsOnline] = useState(
    typeof navigator !== 'undefined' ? navigator.onLine : true
  );

  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 5000);
    const onOnline = () => setIsOnline(true);
    const onOffline = () => setIsOnline(false);
    window.addEventListener('online', onOnline);
    window.addEventListener('offline', onOffline);
    return () => {
      clearInterval(id);
      window.removeEventListener('online', onOnline);
      window.removeEventListener('offline', onOffline);
    };
  }, []);

  const meta = useMemo(
    () =>
      computeDataFreshnessMeta({
        isOnline,
        isSyncing,
        lastSyncAt,
        realtimeEnabled,
        realtimeConnected,
        realtimeDegraded,
        sourceLabel,
        now,
      }),
    [isOnline, isSyncing, lastSyncAt, now, realtimeEnabled, realtimeConnected, realtimeDegraded, sourceLabel]
  );

  return (
    <div className={`${styles.badge} ${styles[meta.level]} ${className}`.trim()} role="status" aria-live="polite">
      <span className={styles.dot} aria-hidden />
      <span>{meta.text}</span>
    </div>
  );
}
