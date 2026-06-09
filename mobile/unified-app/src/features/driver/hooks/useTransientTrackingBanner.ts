import { useEffect, useRef, useState } from "react";

const AUTO_DISMISS_MS = 8_000;

/**
 * Affiche une alerte une seule fois par occurrence de condition, puis la masque
 * automatiquement (toast non bloquant).
 */
export function useTransientTrackingBanner(
  wantsBanner: boolean,
  bannerKind: string | null
): boolean {
  const [visible, setVisible] = useState(false);
  const dismissedKeyRef = useRef<string | null>(null);
  const alertKey = wantsBanner && bannerKind ? bannerKind : null;

  useEffect(() => {
    if (!alertKey) {
      dismissedKeyRef.current = null;
      setVisible(false);
      return;
    }

    if (dismissedKeyRef.current === alertKey) {
      setVisible(false);
      return;
    }

    setVisible(true);
    const timer = setTimeout(() => {
      dismissedKeyRef.current = alertKey;
      setVisible(false);
    }, AUTO_DISMISS_MS);

    return () => clearTimeout(timer);
  }, [alertKey]);

  return visible;
}
