// src/hooks/useDriverLocation.js
import { useState, useEffect } from 'react';

const useDriverLocation = (onLocationUpdate, options) => {
  const [location, setLocation] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!navigator.geolocation) {
      setError("La géolocalisation n'est pas supportée par votre navigateur.");
      return;
    }

    const watchId = navigator.geolocation.watchPosition(
      (position) => {
        const { latitude, longitude } = position.coords;
        const newLocation = { lat: latitude, lng: longitude };
        setLocation(newLocation);
        if (typeof onLocationUpdate === 'function') {
          onLocationUpdate(newLocation);
        }
      },
      (err) => {
        setError(err.message);
      },
      options || { enableHighAccuracy: true, maximumAge: 0, timeout: 10000 }
    );

    // Nettoyage lors du démontage du composant
    return () => {
      navigator.geolocation.clearWatch(watchId);
    };
  }, [onLocationUpdate, options]);

  return { location, error };
};

export default useDriverLocation;
