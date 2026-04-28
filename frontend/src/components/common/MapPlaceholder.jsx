import React, { useEffect, useState } from 'react';
import { useGoogleMapsLoaded } from './GoogleMapsProvider';

const BASE = {
  position: 'relative',
  display: 'flex',
  flexDirection: 'column',
  width: '100%',
  height: '100%',
  minHeight: 200,
  borderRadius: 12,
  overflow: 'hidden',
  fontFamily: "Inter, -apple-system, 'Segoe UI', sans-serif",
  border: '1px solid #E8EEF3',
  background: '#EEF2F6',
};

const SHIMMER_SHEET = {
  position: 'absolute',
  inset: 0,
  background: 'linear-gradient(90deg, #E2E8F0 0%, #F1F5F9 40%, #E8EDF3 60%, #E2E8F0 100%)',
  backgroundSize: '200% 100%',
  animation: 'mapShimmer 1.25s ease-in-out infinite',
};

const FOOTER_HINT = {
  position: 'absolute',
  bottom: 0,
  left: 0,
  right: 0,
  padding: '10px 12px',
  background: 'linear-gradient(to top, rgba(255,255,255,0.86) 0%, rgba(255,255,255,0.95) 100%)',
  borderTop: '1px solid rgba(226, 232, 240, 0.9)',
  fontSize: 11,
  color: '#94A3B8',
  fontWeight: 500,
  letterSpacing: '0.01em',
  transition: 'opacity 0.35s ease',
};

const errorLabel = {
  fontSize: 13,
  color: '#64748B',
  fontWeight: 500,
  textAlign: 'center',
  maxWidth: 280,
  lineHeight: 1.5,
  zIndex: 1,
  padding: '0 12px',
};

/**
 * Espace réservé carte — squelette + léger délai du libellé (évite effet « bug de chargement »
 * quand le SDK est déjà en cache).
 */
export default function MapPlaceholder({ style, /** ms avant d’afficher le texte */ delayLabelMs = 400 }) {
  const { isLoaded, loadError } = useGoogleMapsLoaded();
  const [showLabel, setShowLabel] = useState(false);

  useEffect(() => {
    if (isLoaded && !loadError) {
      setShowLabel(false);
      return undefined;
    }
    setShowLabel(false);
    const t = window.setTimeout(() => setShowLabel(true), delayLabelMs);
    return () => window.clearTimeout(t);
  }, [isLoaded, loadError, delayLabelMs]);

  if (isLoaded && !loadError) return null;

  return (
    <>
      <style>
        {`
        @keyframes mapShimmer {
          0% { background-position: 200% 0; }
          100% { background-position: -200% 0; }
        }
        @media (prefers-reduced-motion: reduce) {
          [data-map-shimmer] {
            animation: none !important;
            background: #E8EDF2 !important;
          }
        }
      `}
      </style>
      <div
        style={{ ...BASE, ...style }}
        role="status"
        data-map-skeleton
        aria-busy={!loadError}
        aria-label="Préparation de la carte"
      >
        {loadError ? (
          <div
            style={{
              flex: 1,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              gap: 10,
              padding: 16,
            }}
          >
            <div
              style={{
                width: 36,
                height: 36,
                borderRadius: '50%',
                background: 'linear-gradient(135deg, #f97316 0%, #ea580c 100%)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: '#fff',
                fontSize: 16,
                fontWeight: 700,
                flexShrink: 0,
              }}
              aria-hidden
            >
              !
            </div>
            <span style={errorLabel}>
              La carte n&apos;a pas pu être chargée. Vérifiez votre connexion et réessayez.
            </span>
            {process.env.NODE_ENV === 'development' && loadError?.message ? (
              <span style={{ ...errorLabel, fontSize: 12, opacity: 0.85 }}>{loadError.message}</span>
            ) : null}
          </div>
        ) : (
          <>
            <div style={SHIMMER_SHEET} aria-hidden data-map-shimmer />
            <div
              style={{
                ...FOOTER_HINT,
                opacity: showLabel ? 1 : 0,
                pointerEvents: 'none',
              }}
            >
              Préparation de la vue carte
            </div>
          </>
        )}
      </div>
    </>
  );
}
