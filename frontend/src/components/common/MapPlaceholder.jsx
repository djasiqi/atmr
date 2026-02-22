import React from 'react';
import { useGoogleMapsLoaded } from './GoogleMapsProvider';

const STYLES = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    width: '100%',
    height: '100%',
    minHeight: 200,
    background: '#f8fafb',
    borderRadius: 12,
    border: '2px solid #E2E8F0',
    gap: 12,
    fontFamily: "Inter, -apple-system, 'Segoe UI', sans-serif",
  },
  icon: {
    width: 40,
    height: 40,
    borderRadius: '50%',
    background: 'linear-gradient(135deg, #00796B 0%, #00695C 100%)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    color: '#fff',
    fontSize: 18,
  },
  spinner: {
    width: 40,
    height: 40,
    border: '3px solid #E2E8F0',
    borderTop: '3px solid #00796B',
    borderRadius: '50%',
    animation: 'mapSpinnerRotate 1s linear infinite',
  },
  label: {
    fontSize: 13,
    color: '#64748B',
    fontWeight: 500,
  },
  errorLabel: {
    fontSize: 13,
    color: '#64748B',
    fontWeight: 500,
    textAlign: 'center',
    maxWidth: 260,
    lineHeight: 1.5,
  },
};

/**
 * Placeholder pour carte Google Maps — loading + error.
 * Utilise automatiquement le contexte GoogleMapsProvider.
 */
export default function MapPlaceholder({ style }) {
  const { isLoaded, loadError } = useGoogleMapsLoaded();

  if (isLoaded && !loadError) return null;

  return (
    <>
      <style>{`@keyframes mapSpinnerRotate { to { transform: rotate(360deg); } }`}</style>
      <div style={{ ...STYLES.container, ...style }}>
        {loadError ? (
          <>
            <div style={STYLES.icon}>!</div>
            <span style={STYLES.errorLabel}>
              La carte n'a pas pu etre chargée. Vérifiez votre connexion et réessayez.
            </span>
          </>
        ) : (
          <>
            <div style={STYLES.spinner} />
            <span style={STYLES.label}>Chargement de la carte...</span>
          </>
        )}
      </div>
    </>
  );
}
