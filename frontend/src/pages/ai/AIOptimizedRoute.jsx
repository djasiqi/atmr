import React, { useState, useMemo } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import apiClient from "../../utils/apiClient";
import { MapContainer, TileLayer, Polyline, Marker, Popup } from "react-leaflet";
import L from "leaflet";
import polyline from "@mapbox/polyline"; // ✅ corrigé
import "./AIOptimizedRoute.css";

// Icône Leaflet par défaut corrigée (sinon marqueurs invisibles dans bundlers)
import "leaflet/dist/leaflet.css";
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl:
    "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
  iconUrl:
    "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
  shadowUrl:
    "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
});

const DEFAULT_CENTER = { lat: 46.2044, lng: 6.1432 }; // Genève
const DEFAULT_ZOOM = 12;

const AIOptimizedRoute = () => {
  const [pickup, setPickup] = useState("");
  const [dropoff, setDropoff] = useState("");
  const [routeLatLngs, setRouteLatLngs] = useState([]); // [[lat, lng], ...]
  const [error, setError] = useState(null);

  // Appel API pour obtenir un itinéraire optimisé
  const optimizeRoute = useMutation({
    mutationFn: async () => {
      const { data } = await apiClient.post("/ai/optimized-route", {
        pickup,
        dropoff,
      });
      return data;
    },
    onSuccess: (data) => {
      try {
        // On accepte plusieurs formats possibles en sortie backend:
        // 1) { polyline: "<encoded>" }
        // 2) { route: { polyline: "<encoded>" } }
        // 3) { route: { coordinates: [[lng,lat], ...] } } (GeoJSON-like)
        // 4) { route: [[lat,lng], ...] }

        let latlngs = [];

        if (data?.polyline) {
          latlngs = polyline.decode(data.polyline).map(([lat, lng]) => [lat, lng]);
        } else if (data?.route?.polyline) {
          latlngs = polyline.decode(data.route.polyline).map(([lat, lng]) => [lat, lng]);
        } else if (Array.isArray(data?.route)) {
          // supposé déjà en [lat,lng]
          latlngs = data.route;
        } else if (data?.route?.coordinates && Array.isArray(data.route.coordinates)) {
          // GeoJSON LineString: [ [lng,lat], ... ] → convertit vers [lat,lng]
          latlngs = data.route.coordinates.map(([lng, lat]) => [lat, lng]);
        } else if (data?.geometry?.coordinates) {
          // Autre cas GeoJSON direct
          latlngs = data.geometry.coordinates.map(([lng, lat]) => [lat, lng]);
        }

        if (!latlngs?.length) throw new Error("Format d'itinéraire inconnu");

        setRouteLatLngs(latlngs);
        setError(null);
      } catch (e) {
        setRouteLatLngs([]);
        setError("Impossible d'interpréter l'itinéraire renvoyé par l’API.");
      }
    },
    onError: () => {
      setError("Erreur lors de la récupération de l'itinéraire.");
      setRouteLatLngs([]);
    },
  });

  // Récupération des trajets recommandés (inchangé)
  const { data: recommendedRoutes, refetch } = useQuery({
    queryKey: ["recommendedRoutes", 1], // ID de l'entreprise
    queryFn: async () => {
      const response = await apiClient.get("/ai/recommended-routes/1");
      return response.data.recommended_routes;
    },
    enabled: false,
  });

  const handleOptimizeRoute = () => {
    if (!pickup || !dropoff) {
      setError("Veuillez saisir les adresses de départ et d'arrivée.");
      return;
    }
    setError(null);
    optimizeRoute.mutate();
  };

  const bounds = useMemo(() => {
    if (!routeLatLngs.length) return null;
    return L.latLngBounds(routeLatLngs.map(([lat, lng]) => [lat, lng]));
  }, [routeLatLngs]);

  return (
    <div className="ai-route-container">
      <h2>🚀 Optimisation d'itinéraire (Leaflet + OSRM)</h2>

      <div className="input-container">
        <input
          type="text"
          placeholder="Adresse de départ"
          value={pickup}
          onChange={(e) => setPickup(e.target.value)}
        />
        <input
          type="text"
          placeholder="Adresse d'arrivée"
          value={dropoff}
          onChange={(e) => setDropoff(e.target.value)}
        />
        <button onClick={handleOptimizeRoute} disabled={optimizeRoute.isPending}>
          {optimizeRoute.isPending ? "Calcul..." : "Optimiser le trajet"}
        </button>
      </div>

      {error && <p className="error">{error}</p>}

      <div className="map-container">
        <MapContainer
          center={DEFAULT_CENTER}
          zoom={DEFAULT_ZOOM}
          style={{ width: "100%", height: "400px" }}
          bounds={bounds || undefined}
        >
          <TileLayer
            attribution='&copy; OpenStreetMap contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />

          {/* Marqueurs pickup/dropoff si l’utilisateur entre des coordonnées lat,lng */}
          {/^[-+]?\d+(\.\d+)?,\s*[-+]?\d+(\.\d+)?$/.test(pickup) && (() => {
            const [lat, lng] = pickup.split(",").map(Number);
            return (
              <Marker position={[lat, lng]}>
                <Popup>Départ</Popup>
              </Marker>
            );
          })()}

          {/^[-+]?\d+(\.\d+)?,\s*[-+]?\d+(\.\d+)?$/.test(dropoff) && (() => {
            const [lat, lng] = dropoff.split(",").map(Number);
            return (
              <Marker position={[lat, lng]}>
                <Popup>Arrivée</Popup>
              </Marker>
            );
          })()}

          {routeLatLngs.length > 0 && (
            <Polyline positions={routeLatLngs} />
          )}
        </MapContainer>
      </div>

      <h3>📊 Trajets les plus fréquents</h3>
      <button onClick={() => refetch()}>
        🔄 Mettre à jour les recommandations
      </button>
      {recommendedRoutes && (
        <ul>
          {recommendedRoutes.map((route, index) => (
            <li key={index}>
              {route[0]} ({route[1]} fois)
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

export default AIOptimizedRoute;
