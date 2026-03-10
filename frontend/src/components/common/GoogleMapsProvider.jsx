import React, { createContext, useContext, useState, useEffect, useRef, Component } from 'react';

const LIBRARIES = ['places', 'geometry'];
const GOOGLE_MAPS_SCRIPT_ID = 'google-maps-script';

const GoogleMapsContext = createContext({ isLoaded: false, loadError: null });

const isSdkReady = () =>
  typeof window !== 'undefined' &&
  typeof window.google?.maps?.Map === 'function';

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
  const [isLoaded, setIsLoaded] = useState(isSdkReady);
  const [loadError, setLoadError] = useState(null);
  const attemptedRef = useRef(false);

  useEffect(() => {
    if (isLoaded || attemptedRef.current) return;
    attemptedRef.current = true;

    if (isSdkReady()) {
      setIsLoaded(true);
      return;
    }

    const apiKey = process.env.REACT_APP_GOOGLE_MAPS_API_KEY;
    if (!apiKey) {
      console.warn('[GoogleMaps] REACT_APP_GOOGLE_MAPS_API_KEY non définie');
      setLoadError(new Error('API key manquante'));
      return;
    }

    // Vérifier si le script existe déjà (double-render StrictMode)
    let script = document.getElementById(GOOGLE_MAPS_SCRIPT_ID);
    if (script) {
      const check = () => {
        if (isSdkReady()) setIsLoaded(true);
        else setTimeout(check, 100);
      };
      check();
      return;
    }

    // Charger le script avec loading=async (recommandation Google)
    const libs = LIBRARIES.join(',');
    script = document.createElement('script');
    script.id = GOOGLE_MAPS_SCRIPT_ID;
    script.src = `https://maps.googleapis.com/maps/api/js?key=${apiKey}&libraries=${libs}&loading=async`;
    script.async = true;
    script.defer = true;

    script.onload = () => {
      const check = () => {
        if (isSdkReady()) {
          setIsLoaded(true);
        } else {
          setTimeout(check, 50);
        }
      };
      check();
    };

    script.onerror = () => {
      console.warn('[GoogleMaps] Erreur chargement SDK');
      setLoadError(new Error('Échec chargement Google Maps SDK'));
    };

    document.head.appendChild(script);
  }, [isLoaded]);

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
