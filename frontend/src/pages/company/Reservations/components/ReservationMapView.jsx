import React, { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import styles from './ReservationMapView.module.css';

const ReservationMapView = ({ reservations }) => {
  const mapRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const markersLayerRef = useRef(null); // Groupe de marqueurs pour pouvoir les nettoyer
  const [geocodingStatus, setGeocodingStatus] = useState('idle');

  // Déterminer la date affichée (première réservation ou aujourd'hui)
  const displayDate = useMemo(() => {
    if (reservations.length > 0) {
      const firstReservation = reservations[0];
      const date = new Date(firstReservation.scheduled_time || firstReservation.pickup_time);
      return date.toLocaleDateString('fr-FR', {
        weekday: 'long',
        day: 'numeric',
        month: 'long',
        year: 'numeric',
      });
    }
    // Si pas de réservations, afficher aujourd'hui
    return new Date().toLocaleDateString('fr-FR', {
      weekday: 'long',
      day: 'numeric',
      month: 'long',
      year: 'numeric',
    });
  }, [reservations]);

  // Fonction de géocodification simple avec Nominatim
  const geocodeAddress = useCallback(async (address) => {
    try {
      const response = await fetch(
        `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(
          address
        )}&limit=1&countrycodes=ch`
      );
      const data = await response.json();
      if (data.length > 0) {
        return [parseFloat(data[0].lat), parseFloat(data[0].lon)];
      }
    } catch (error) {
      console.error('Erreur de géocodification:', error);
    }
    return null;
  }, []);

  // Fonction pour obtenir l'itinéraire OSRM via le backend
  const getOSRMRoute = useCallback(async (pickupCoords, dropoffCoords) => {
    try {
      // Essayer d'utiliser le backend API qui a accès à OSRM
      const url =
        `${process.env.REACT_APP_API_URL || 'http://localhost:3000/api'}/osrm/route?` +
        `pickup_lat=${pickupCoords[0]}&pickup_lon=${pickupCoords[1]}&` +
        `dropoff_lat=${dropoffCoords[0]}&dropoff_lon=${dropoffCoords[1]}`;

      const response = await fetch(url);

      if (response.ok) {
        const data = await response.json();

        if (data.route && data.route.length > 0) {
          return data.route;
        }
      } else {
        // eslint-disable-next-line no-console
        console.error('❌ Erreur HTTP:', response.status, await response.text());
      }
    } catch (error) {
      console.error('❌ OSRM erreur:', error);
    }

    // Fallback : retourner une ligne droite
    console.warn('⚠️ Fallback: ligne droite');
    return [pickupCoords, dropoffCoords];
  }, []);

  // Initialiser la carte (une seule fois)
  useEffect(() => {
    if (!mapRef.current || mapInstanceRef.current) return;

    // Initialiser la carte
    const map = L.map(mapRef.current).setView([46.2044, 6.1432], 12);
    mapInstanceRef.current = map;

    // Créer un groupe de couches pour les marqueurs
    const markersLayer = L.layerGroup().addTo(map);
    markersLayerRef.current = markersLayer;

    // Ajouter les tuiles
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '© OpenStreetMap contributors',
    }).addTo(map);

    return () => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove();
        mapInstanceRef.current = null;
        markersLayerRef.current = null;
      }
    };
  }, []);

  // Mettre à jour les marqueurs quand les réservations changent
  useEffect(() => {
    if (!mapInstanceRef.current || !markersLayerRef.current) return;

    const map = mapInstanceRef.current;
    const markersLayer = markersLayerRef.current;
    let isCancelled = false;

    // Nettoyer les marqueurs existants
    markersLayer.clearLayers();

    // Ajouter les marqueurs pour chaque réservation
    const addMarkers = async () => {
      if (isCancelled) return;
      setGeocodingStatus('loading');
      let markersCount = 0;

      for (const reservation of reservations) {
        if (isCancelled || !mapInstanceRef.current) break;

        // Essayer différentes sources de coordonnées
        let pickupCoords = null;
        let dropoffCoords = null;

        // Coordonnées de prise en charge
        const pickupLat = reservation.pickup_lat || reservation.pickupLat;
        const pickupLon = reservation.pickup_lon || reservation.pickupLon;

        if (pickupLat && pickupLon && !isNaN(pickupLat) && !isNaN(pickupLon)) {
          pickupCoords = [parseFloat(pickupLat), parseFloat(pickupLon)];
        } else if (reservation.pickup_location) {
          // Géocodification si pas de coordonnées
          pickupCoords = await geocodeAddress(reservation.pickup_location);
        }

        // Coordonnées de destination
        const dropoffLat = reservation.dropoff_lat || reservation.dropoffLat;
        const dropoffLon = reservation.dropoff_lon || reservation.dropoffLon;

        if (dropoffLat && dropoffLon && !isNaN(dropoffLat) && !isNaN(dropoffLon)) {
          dropoffCoords = [parseFloat(dropoffLat), parseFloat(dropoffLon)];
        } else if (reservation.dropoff_location) {
          // Géocodification si pas de coordonnées
          dropoffCoords = await geocodeAddress(reservation.dropoff_location);
        }

        // Vérifier que la carte existe toujours avant d'ajouter les marqueurs
        if (isCancelled || !mapInstanceRef.current) break;

        // Créer des icônes personnalisées avec emojis
        const pickupIcon = L.divIcon({
          html: '<div style="font-size: 32px; text-align: center; line-height: 1;">📍</div>',
          className: 'custom-marker-icon',
          iconSize: [32, 32],
          iconAnchor: [16, 32],
          popupAnchor: [0, -32],
        });

        const dropoffIcon = L.divIcon({
          html: '<div style="font-size: 32px; text-align: center; line-height: 1;">🎯</div>',
          className: 'custom-marker-icon',
          iconSize: [32, 32],
          iconAnchor: [16, 32],
          popupAnchor: [0, -32],
        });

        // Créer le marqueur de destination (caché par défaut, ne l'ajoute pas à la carte)
        let dropoffMarker = null;
        if (dropoffCoords && !isCancelled && mapInstanceRef.current) {
          dropoffMarker = L.marker(dropoffCoords, {
            icon: dropoffIcon,
          });
          dropoffMarker.bindPopup(`
            <div class="${styles.popupContent}">
              <h4>🎯 Destination</h4>
              <p><strong>Client:</strong> ${
                reservation.customer_name || reservation.client?.full_name || 'N/A'
              }</p>
              <p><strong>Adresse:</strong> ${reservation.dropoff_location}</p>
              <p><strong>Montant:</strong> ${Number(reservation.amount || 0).toFixed(2)} CHF</p>
              <p><strong>ID:</strong> #${reservation.id}</p>
            </div>
          `);
        }

        // Ajouter le marqueur de prise en charge avec logique d'affichage
        if (pickupCoords) {
          try {
            const pickupMarker = L.marker(pickupCoords, {
              icon: pickupIcon,
            }).addTo(markersLayer); // Ajouter au groupe plutôt qu'à la carte
            pickupMarker.bindPopup(`
              <div class="${styles.popupContent}">
                <h4>📍 Prise en charge</h4>
                <p><strong>Client:</strong> ${
                  reservation.customer_name || reservation.client?.full_name || 'N/A'
                }</p>
                <p><strong>Adresse:</strong> ${reservation.pickup_location}</p>
                <p><strong>Heure:</strong> ${new Date(
                  reservation.scheduled_time
                ).toLocaleTimeString('fr-FR')}</p>
                <p><strong>Statut:</strong> ${reservation.status}</p>
                <p><strong>ID:</strong> #${reservation.id}</p>
              </div>
            `);
            markersCount++;

            // Variable pour stocker la polyline
            let routePolyline = null;

            // Événement lors du clic sur le marqueur de prise en charge
            pickupMarker.on('click', async () => {
              if (dropoffCoords && dropoffMarker) {
                // Afficher le marqueur de destination
                if (!markersLayer.hasLayer(dropoffMarker)) {
                  dropoffMarker.addTo(markersLayer);
                }

                // Obtenir et afficher l'itinéraire réel
                if (!routePolyline && pickupCoords && dropoffCoords) {
                  const route = await getOSRMRoute(pickupCoords, dropoffCoords);
                  if (route && route.length > 0) {
                    // Déterminer la couleur selon le statut
                    let lineColor = '#00796b';
                    switch (reservation.status) {
                      case 'pending':
                        lineColor = '#ff9800';
                        break;
                      case 'accepted':
                      case 'assigned':
                        lineColor = '#2196f3';
                        break;
                      case 'in_progress':
                        lineColor = '#00796b';
                        break;
                      case 'completed':
                        lineColor = '#4caf50';
                        break;
                      case 'canceled':
                        lineColor = '#f44336';
                        break;
                      default:
                        lineColor = '#00796b';
                    }

                    routePolyline = L.polyline(route, {
                      color: lineColor,
                      weight: 4,
                      opacity: 0.7,
                    }).addTo(map);

                    // Ajuster la vue pour voir tout l'itinéraire
                    const bounds = L.latLngBounds([pickupCoords, dropoffCoords]);
                    map.fitBounds(bounds, { padding: [50, 50] });
                  }
                }
              }
            });

            // Événement lors de la fermeture du popup
            pickupMarker.on('popupclose', () => {
              // Cacher le marqueur de destination
              if (dropoffMarker && markersLayer.hasLayer(dropoffMarker)) {
                markersLayer.removeLayer(dropoffMarker);
              }
              // Supprimer l'itinéraire
              if (routePolyline && map.hasLayer(routePolyline)) {
                map.removeLayer(routePolyline);
                routePolyline = null;
              }
            });
          } catch (error) {
            console.error("Erreur lors de l'ajout du marqueur de prise en charge:", error);
          }
        }
      }

      if (!isCancelled) {
        setGeocodingStatus(markersCount > 0 ? 'success' : 'no-data');
      }
    };

    addMarkers();

    return () => {
      isCancelled = true;
    };
  }, [reservations, geocodeAddress, getOSRMRoute]);

  return (
    <div className={styles.mapContainer}>
      <div ref={mapRef} className={styles.map}></div>

      {/* Message d'état */}
      {geocodingStatus === 'loading' && (
        <div className={styles.statusMessage}>
          <span>🔄 Chargement des positions...</span>
        </div>
      )}

      {geocodingStatus === 'no-data' && (
        <div className={styles.statusMessage}>
          <span>📍 Aucune réservation pour cette journée</span>
        </div>
      )}

      {reservations.length === 0 && geocodingStatus === 'idle' && (
        <div className={styles.statusMessage}>
          <span>📭 Aucune réservation à afficher</span>
        </div>
      )}

      {/* Info sur les réservations affichées - Seulement si > 0 */}
      {reservations.length > 0 && (
        <div className={styles.mapInfo}>
          <div className={styles.mapInfoRow}>
            <span className={styles.mapInfoIcon}>📅</span>
            <span className={styles.mapInfoText}>{displayDate}</span>
          </div>
          <div className={styles.mapInfoRow}>
            <span className={styles.mapInfoIcon}>🗺️</span>
            <span className={styles.mapInfoText}>
              {reservations.length} réservation{reservations.length > 1 ? 's' : ''}
            </span>
          </div>
        </div>
      )}

      <div className={styles.mapLegend}>
        <div className={styles.legendItem}>
          <span className={styles.legendIcon}>📍</span>
          <span>Prise en charge</span>
        </div>
        <div className={styles.legendItem}>
          <span className={styles.legendIcon}>🎯</span>
          <span>Destination</span>
        </div>
      </div>
    </div>
  );
};

export default ReservationMapView;
