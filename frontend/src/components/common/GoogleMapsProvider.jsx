import React, {
  createContext,
  useContext,
  useState,
  useEffect,
  useLayoutEffect,
  Component,
} from 'react';

import { isGoogleMapsSdkReady, loadGoogleMapsScript } from '../../utils/googleMapsLoader';

const GoogleMapsContext = createContext({ isLoaded: false, loadError: null });

/**
 * Hook pour savoir si Google Maps SDK est chargé.
 * @returns {{ isLoaded: boolean, loadError: Error | null }}
 */
export function useGoogleMapsLoaded() {
  return useContext(GoogleMapsContext);
}

/**
 * Charge Google Maps manuellement via <script> — plus robuste
 * que useJsApiLoader en StrictMode et avec HMR.
 */
function GoogleMapsLoader({ children }) {
  const [isLoaded, setIsLoaded] = useState(isGoogleMapsSdkReady);
  const [loadError, setLoadError] = useState(null);
  const isDev = process.env.NODE_ENV !== 'production';

  // Même frame que le premier paint : si le SDK est déjà en cache, évite un flash « non chargé ».
  useLayoutEffect(() => {
    if (isGoogleMapsSdkReady()) {
      setIsLoaded(true);
    }
  }, []);

  useEffect(() => {
    if (isLoaded) return;

    if (isGoogleMapsSdkReady()) {
      setIsLoaded(true);
      return;
    }

    const apiKey = process.env.REACT_APP_GOOGLE_MAPS_API_KEY;
    if (!apiKey) {
      console.warn('[GoogleMaps] REACT_APP_GOOGLE_MAPS_API_KEY non définie');
      setLoadError(new Error('API key manquante'));
      return;
    }

    let cancelled = false;
    let timeoutId = null;
    let idleId = null;

    const run = () => {
      if (cancelled) return;
      if (isDev && window.google?.maps && typeof window.google.maps.importLibrary !== 'function') {
        console.info('[GoogleMaps] SDK détecté, finalisation importLibrary en cours...');
      }
      loadGoogleMapsScript()
        .then(() => {
          if (!cancelled) setIsLoaded(true);
        })
        .catch((e) => {
          if (!cancelled) {
            if (isDev && /importLibrary indisponible/i.test(String(e?.message || ''))) {
              console.warn(
                '[GoogleMaps] Échec finalisation SDK : importLibrary indisponible après timeout'
              );
            }
            console.warn('[GoogleMaps] Chargement SDK', e?.message);
            setLoadError(e instanceof Error ? e : new Error(String(e)));
          }
        });
    };

    const isLanding = typeof window !== 'undefined' && window.location?.pathname === '/';
    if (!isLanding) {
      run();
    } else if (typeof window.requestIdleCallback === 'function') {
      idleId = window.requestIdleCallback(() => run(), { timeout: 2500 });
      timeoutId = window.setTimeout(() => run(), 2800);
    } else {
      timeoutId = window.setTimeout(() => run(), 1200);
    }

    return () => {
      cancelled = true;
      if (timeoutId) window.clearTimeout(timeoutId);
      if (idleId && typeof window.cancelIdleCallback === 'function') {
        window.cancelIdleCallback(idleId);
      }
    };
  }, [isLoaded, isDev]);

  return (
    <GoogleMapsContext.Provider value={{ isLoaded, loadError }}>
      {children}
    </GoogleMapsContext.Provider>
  );
}

/**
 * Error boundary — si le chargement crash,
 * l'app continue sans cartes.
 */
class GoogleMapsErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  componentDidCatch(error) {
    console.warn('[GoogleMaps] Crash intercepté, cartes désactivées:', error.message);
  }

  render() {
    if (this.state.hasError) {
      return (
        <GoogleMapsContext.Provider value={{ isLoaded: false, loadError: new Error('SDK crash') }}>
          {this.props.children}
        </GoogleMapsContext.Provider>
      );
    }
    return this.props.children;
  }
}

/**
 * Provider centralisé Google Maps — crash-proof.
 * Charge le SDK manuellement (pas useJsApiLoader) pour éviter
 * les crashes en StrictMode / HMR.
 * L'app reste 100% fonctionnelle meme si Google Maps échoue.
 */
export default function GoogleMapsProvider({ children }) {
  return (
    <GoogleMapsErrorBoundary>
      <GoogleMapsLoader>
        {children}
      </GoogleMapsLoader>
    </GoogleMapsErrorBoundary>
  );
}
