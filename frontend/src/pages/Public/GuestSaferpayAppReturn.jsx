import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { useSearchParams } from 'react-router-dom';

const APP_SCHEME = 'lirie';
/** Aligner avec mobile/unified-app app.json `android.package` (ouverture intent) */
const ANDROID_PACKAGE = 'ch.lirie.app';

/**
 * Ouvre l’appli (schéma lirie). Les navigateurs intégrés (Safari View, Custom Tabs) bloquent
 * souvent location.replace ; le liant utilisateur + fermeture du Paiement dans l’app reste
 * le scénario fiable (Expo ferme l’in-app, puis l’appli enchaîne sur /guest-payment-return).
 */
export default function GuestSaferpayAppReturn() {
  const [searchParams] = useSearchParams();
  const [copied, setCopied] = useState(false);
  const q = searchParams.toString();

  const { lirieUrl, androidIntentUrl } = useMemo(() => {
    const lirie = q ? `${APP_SCHEME}://guest-payment-return?${q}` : `${APP_SCHEME}://guest-payment-return`;
    const pathAndQuery = q
      ? `//guest-payment-return?${q}`
      : `//guest-payment-return`;
    const intent = `intent:${pathAndQuery}#Intent;scheme=${APP_SCHEME};package=${ANDROID_PACKAGE};end`;
    return { lirieUrl: lirie, androidIntentUrl: intent };
  }, [q]);

  const isAndroid = typeof navigator !== 'undefined' && /android/i.test(navigator.userAgent);

  const tryOpen = useCallback(() => {
    if (isAndroid) {
      window.location.assign(androidIntentUrl);
      return;
    }
    window.location.assign(lirieUrl);
  }, [isAndroid, androidIntentUrl, lirieUrl]);

  useEffect(() => {
    // Une tentative (souvent ignorée en in-app) ; le bouton reste la repli.
    tryOpen();
  }, [tryOpen]);

  const onCopy = async () => {
    try {
      await navigator.clipboard.writeText(lirieUrl);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // ignore
    }
  };

  return (
    <div
      style={{
        minHeight: '100vh',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        padding: 24,
        fontFamily: 'system-ui, sans-serif',
        textAlign: 'center',
        background: '#EAF3F1',
        color: '#0f172a',
      }}
    >
      <p style={{ maxWidth: 420, lineHeight: 1.6, fontSize: 16 }}>
        <strong>Le plus fiable</strong> : après le paiement, fermez l’écran bancier (
        <strong>Terminé</strong> ou <strong>✕</strong> en haut) — vous revenez dans l’appli
        LIRiE et la vérification du paiement se lance automatiquement. Les vues
        bancaires intégrées bloquent souvent les liens <code>lirie://</code> (le bouton peut
        alors ne rien faire).
      </p>
      <a
        href={isAndroid ? androidIntentUrl : lirieUrl}
        rel="noopener noreferrer"
        style={{
          marginTop: 20,
          display: 'inline-block',
          padding: '14px 24px',
          background: '#0A8F7A',
          color: '#fff',
          border: 'none',
          borderRadius: 12,
          fontWeight: 600,
          fontSize: 16,
          cursor: 'pointer',
          textDecoration: 'none',
        }}
      >
        {isAndroid ? 'Ouvrir dans l’appli (Android)' : 'Ouvrir dans l’appli'}
      </a>
      <a
        href={isAndroid ? androidIntentUrl : lirieUrl}
        onClick={() => {
          // Laisse le navigateur gérer (geste utilisateur)
        }}
        style={{ marginTop: 14, color: '#0A8F7A', fontWeight: 500 }}
        rel="noopener noreferrer"
      >
        Lien direct
      </a>
      <button
        type="button"
        onClick={onCopy}
        style={{
          marginTop: 12,
          background: 'transparent',
          color: '#475569',
          border: '1px solid #94a3b8',
          borderRadius: 8,
          padding: '8px 14px',
          cursor: 'pointer',
        }}
      >
        {copied ? 'Lien copié' : 'Copier le lien d’ouverture'}
      </button>
    </div>
  );
}
