import React, { useEffect, useMemo, useState } from "react";
import apiClient from "../../../utils/apiClient";
import { useParams, useNavigate } from "react-router-dom";
import "./ClientDashboard.css";
import { useMutation } from "@tanstack/react-query";

// 🔁 Leaflet (remplace Google Maps)
import {
  MapContainer,
  TileLayer,
  Polyline,
  Marker,
  Popup,
  useMap,
} from "react-leaflet";
import L from "leaflet";
import polyline from "@mapbox/polyline";

// UI
import HeaderDashboard from "../../../components/layout/Header/HeaderDashboard";
import Footer from "../../../components/layout/Footer/Footer";
import Slider from "react-slick";
import "slick-carousel/slick/slick.css";
import "slick-carousel/slick/slick-theme.css";

// Petit helper pour fitBounds dynamique
const FitBounds = ({ bounds }) => {
  const map = useMap();
  useEffect(() => {
    if (bounds) map.fitBounds(bounds, { padding: [24, 24] });
  }, [bounds, map]);
  return null;
};

const ClientDashboard = () => {
  const { id: clientId } = useParams();
  const navigate = useNavigate();

  // États profil & réservations
  const [profile, setProfile] = useState(null);
  const [loadingProfile, setLoadingProfile] = useState(true);
  const [upcomingBookings, setUpcomingBookings] = useState([]);
  const [ongoingBookings, setOngoingBookings] = useState([]);
  const [pastBookings, setPastBookings] = useState([]);
  const [suggestions, setSuggestions] = useState([]);
  const [error, setError] = useState(null);
  const [loadingBookings, setLoadingBookings] = useState(false);

  // Formulaire
  const [pickup, setPickup] = useState("");
  const [destination, setDestination] = useState("");

  // 🔁 Remplace "directions" Google par des latlngs pour Leaflet
  const [routeLatLngs, setRouteLatLngs] = useState([]); // [[lat,lng], ...]
  const [loadingOptimization, setLoadingOptimization] = useState(false);

  const center = useMemo(() => ({ lat: 46.2044, lng: 6.1432 }), []);

  // ID client effectif (URL ou storage)
  const effectiveClientId = useMemo(() => {
    return clientId || localStorage.getItem("public_id");
  }, [clientId]);

  // Profil client
  useEffect(() => {
    const token = localStorage.getItem("authToken");
    if (!token) {
      navigate("/login");
      return;
    }
    if (!effectiveClientId) {
      setLoadingProfile(false);
      setError("Identifiant client introuvable.");
      return;
    }
    setLoadingProfile(true);
    apiClient
      .get(`/clients/${effectiveClientId}`, {
        headers: { Authorization: `Bearer ${token}` },
      })
      .then((response) => setProfile(response.data))
      .catch((err) => {
        console.error("Erreur profil :", err);
        setError("Impossible de charger le profil utilisateur.");
      })
      .finally(() => setLoadingProfile(false));
  }, [effectiveClientId, navigate]);

  // Mutation: optimisation d’itinéraire (backend IA/OSRM)
  const { mutate: triggerOptimizeRoute, isPending: isOptimizing } = useMutation(
    {
      mutationFn: async () => {
        if (!pickup || !destination) return null;
        setLoadingOptimization(true);
        const response = await apiClient.post("/ai/optimized-route", {
          pickup,
          dropoff: destination,
        });
        return response.data;
      },
      onSuccess: (data) => {
        setLoadingOptimization(false);
        try {
          // On accepte plusieurs formats renvoyés par ton backend:
          // 1) { polyline: "<enc>" }
          // 2) { route: { polyline: "<enc>" } }
          // 3) { route: [[lat,lng], ...] }
          // 4) { route: { coordinates: [[lng,lat], ...] } } (GeoJSON-like)
          // 5) { geometry: { coordinates: [[lng,lat], ...] } }

          let latlngs = [];
          if (data?.polyline) {
            latlngs = polyline
              .decode(data.polyline)
              .map(([lat, lng]) => [lat, lng]);
          } else if (data?.route?.polyline) {
            latlngs = polyline
              .decode(data.route.polyline)
              .map(([lat, lng]) => [lat, lng]);
          } else if (Array.isArray(data?.route)) {
            latlngs = data.route; // supposé [lat,lng]
          } else if (data?.route?.coordinates) {
            latlngs = data.route.coordinates.map(([lng, lat]) => [lat, lng]);
          } else if (data?.geometry?.coordinates) {
            latlngs = data.geometry.coordinates.map(([lng, lat]) => [lat, lng]);
          }

          if (!latlngs.length) throw new Error("Format d'itinéraire inconnu");
          setRouteLatLngs(latlngs);
        } catch (e) {
          console.error("Parsing itinéraire:", e);
          setError("Impossible d'obtenir l'itinéraire optimisé.");
          setRouteLatLngs([]);
        }
      },
      onError: () => {
        setLoadingOptimization(false);
        setError("Erreur lors de la récupération de l'itinéraire.");
        setRouteLatLngs([]);
      },
    }
  );

  // Déclenche auto après 2s si pickup/destination changent
  useEffect(() => {
    if (!pickup || !destination) return;
    const t = setTimeout(() => {
      if (!isOptimizing) triggerOptimizeRoute();
    }, 2000);
    return () => clearTimeout(t);
  }, [pickup, destination, triggerOptimizeRoute, isOptimizing]);

  // Réservations
  useEffect(() => {
    if (!effectiveClientId) return;
    const token = localStorage.getItem("authToken");
    setLoadingBookings(true);
    apiClient
      .get(`/clients/${effectiveClientId}/bookings`, {
        headers: { Authorization: `Bearer ${token}` },
      })
      .then((response) => {
        const bookingsArray = response.data;
        const now = Date.now();
        const upcoming = bookingsArray.filter(
          (b) => Date.parse(b.scheduled_time) > now
        );
        const past = bookingsArray.filter(
          (b) => Date.parse(b.scheduled_time) <= now
        );
        setUpcomingBookings(upcoming);
        setPastBookings(past);
        setOngoingBookings([]);
        const sortedPast = [...past].sort(
          (a, b) => Date.parse(b.scheduled_time) - Date.parse(a.scheduled_time)
        );
        setSuggestions(sortedPast.slice(0, 3).map((b) => b.dropoff_location));
      })
      .catch((err) => {
        console.error("Erreur réservations :", err);
        setError("Impossible de charger les réservations.");
      })
      .finally(() => setLoadingBookings(false));
  }, [effectiveClientId]);

  // Champs médicaux (inchangé)
  const [medicalFacility, setMedicalFacility] = useState("");
  const [doctorName, setDoctorName] = useState("");
  const [showMedicalFields, setShowMedicalFields] = useState(false);

  useEffect(() => {
    if (!destination) return;
    const lower = destination.toLowerCase();
    const medicalKeywords = [
      "hôpital",
      "hopital",
      "hug",
      "ems",
      "cabinet",
      "clinique",
      "médecin",
      "docteur",
    ];
    const doctorKeywords = [
      "docteur",
      "dr",
      "dr.",
      "dr med",
      "dr méd",
      "médecin",
    ];

    const isMedicalFacility = medicalKeywords.some((k) => lower.includes(k));
    const isDoctor = doctorKeywords.some((k) => lower.includes(k));

    if (isDoctor) {
      setDoctorName(destination);
      setShowMedicalFields(true);
    }
    if (isMedicalFacility) {
      setMedicalFacility(destination);
      setShowMedicalFields(true);
    }
  }, [destination]);

  const [selectedDate, setSelectedDate] = useState("");
  const [selectedTime, setSelectedTime] = useState("");

  const handleOptimizeRoute = () => {
    if (
      !pickup ||
      pickup.length < 5 ||
      !destination ||
      destination.length < 5
    ) {
      setError("Veuillez entrer une adresse complète et valide.");
      return;
    }
    setError(null);
    triggerOptimizeRoute();
  };

  const handleBooking = () => {
    const token = localStorage.getItem("authToken");
    if (!token) {
      setError("Token d'authentification manquant.");
      return;
    }
    if (!pickup || !destination) {
      setError("Veuillez saisir le lieu de départ et la destination.");
      return;
    }
    if (!selectedDate || !selectedTime) {
      setError("Veuillez sélectionner une date et une heure.");
      return;
    }

    const scheduledDateTime = new Date(`${selectedDate}T${selectedTime}:00`);
    const scheduledDateTimeUTC = new Date(
      scheduledDateTime.getTime() -
        scheduledDateTime.getTimezoneOffset() * 60000
    );

    const bookingData = {
      pickup_location: pickup,
      dropoff_location: destination,
      scheduled_time: scheduledDateTimeUTC.toISOString(),
      amount: 50,
      medical_facility: medicalFacility,
      doctor_name: doctorName,
    };

    apiClient
      .post(`/clients/${effectiveClientId}/bookings`, bookingData, {
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
        },
      })
      .then((response) => {
        alert("Réservation effectuée avec succès !");
        setUpcomingBookings([...upcomingBookings, response.data.booking]);
        setPickup("");
        setDestination("");
        setSelectedDate("");
        setSelectedTime("");
        setMedicalFacility("");
        setDoctorName("");
      })
      .catch((err) => {
        console.error("Erreur réservation :", err);
        setError("Une erreur est survenue lors de la réservation.");
      });
  };

  return (
    <div className="container">
      {/* Nom affiché robuste (snake/camel/nest) */}
      {(() => {
        const p = profile || {};
        const userName =
          p.first_name ??
          p.firstName ??
          p.username ??
          p.user?.first_name ??
          p.user?.username ??
          "Utilisateur";
        return <HeaderDashboard userName={userName} />;
      })()}

      {loadingProfile && <p>Chargement du profil…</p>}

      {loadingBookings && <p>Chargement des réservations...</p>}
      {error && <p className="error">{error}</p>}

      <main>
        <h1 className="title">Commandez une course</h1>
        <div className="mainRow">
          {/* Section gauche */}
          <div className="leftSection">
            <form className="form">
              <div className="inputWrapper">
                {/* Champ simple (on branchera Photon plus tard) */}
                <input
                  type="text"
                  value={pickup}
                  onChange={(e) => setPickup(e.target.value)}
                  className="input"
                  placeholder="Saisissez votre lieu de départ"
                />
              </div>
              <div className="inputWrapper">
                <input
                  type="text"
                  value={destination}
                  onChange={(e) => setDestination(e.target.value)}
                  className="input"
                  placeholder="Saisissez votre destination"
                />
              </div>

              {showMedicalFields && (
                <>
                  <div className="inputWrapper">
                    <label>Établissement médical</label>
                    <input
                      type="text"
                      value={medicalFacility}
                      onChange={(e) => setMedicalFacility(e.target.value)}
                      className="input"
                      placeholder="Nom de l'établissement"
                    />
                  </div>
                  <div className="inputWrapper">
                    <label>Nom du Docteur</label>
                    <input
                      type="text"
                      value={doctorName}
                      onChange={(e) => setDoctorName(e.target.value)}
                      className="input"
                      placeholder="Nom du médecin"
                    />
                  </div>
                </>
              )}

              <div className="dateTime">
                <input
                  type="date"
                  value={selectedDate}
                  onChange={(e) => setSelectedDate(e.target.value)}
                  className="input"
                />
                <input
                  type="time"
                  value={selectedTime}
                  onChange={(e) => setSelectedTime(e.target.value)}
                  className="input"
                />
              </div>

              {/* Suggestions */}
              <div className="suggestionsSection">
                <h3>Suggestions de destinations</h3>
                {suggestions.length > 0 ? (
                  <div className="sliderWrapper">
                    <Slider
                      dots={false}
                      infinite
                      speed={500}
                      slidesToShow={3}
                      slidesToScroll={1}
                      vertical
                      verticalSwiping
                      centerMode
                      centerPadding="40px"
                      focusOnSelect
                      arrows={false}
                      responsive={[
                        {
                          breakpoint: 1024,
                          settings: {
                            slidesToShow: 3,
                            slidesToScroll: 1,
                            centerPadding: "30px",
                          },
                        },
                        {
                          breakpoint: 768,
                          settings: {
                            slidesToShow: 1,
                            slidesToScroll: 1,
                            centerPadding: "20px",
                          },
                        },
                      ]}
                    >
                      {suggestions.map((dest, index) => (
                        <div key={index} className="suggestionSlide">
                          <div
                            className="suggestionCard"
                            onClick={() => setDestination(dest)}
                          >
                            {dest}
                          </div>
                        </div>
                      ))}
                    </Slider>
                  </div>
                ) : (
                  <p>Aucune suggestion disponible.</p>
                )}
              </div>

              <button
                type="button"
                className="primaryButton"
                onClick={handleOptimizeRoute}
                disabled={loadingOptimization}
              >
                {loadingOptimization
                  ? "Optimisation en cours..."
                  : "🔍 Optimiser mon trajet"}
              </button>

              <button
                type="button"
                className="primaryButton"
                onClick={handleBooking}
              >
                Réserver la course
              </button>
            </form>
          </div>

          {/* Section droite : Carte Leaflet */}
          <div className="rightSection">
            <div style={{ height: "400px", width: "100%" }}>
              <MapContainer
                center={center}
                zoom={12}
                style={{ width: "100%", height: "100%" }}
              >
                <TileLayer
                  attribution="&copy; OpenStreetMap contributors"
                  url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                />

                {/* Affiche des marqueurs si l'utilisateur saisit "lat,lng" */}
                {/^-?\d+(\.\d+)?,\s*-?\d+(\.\d+)?$/.test(pickup) &&
                  (() => {
                    const [lat, lng] = pickup.split(",").map(Number);
                    return (
                      <Marker position={[lat, lng]}>
                        <Popup>Départ</Popup>
                      </Marker>
                    );
                  })()}

                {/^-?\d+(\.\d+)?,\s*-?\d+(\.\d+)?$/.test(destination) &&
                  (() => {
                    const [lat, lng] = destination.split(",").map(Number);
                    return (
                      <Marker position={[lat, lng]}>
                        <Popup>Arrivée</Popup>
                      </Marker>
                    );
                  })()}

                {routeLatLngs.length > 0 && (
                  <Polyline positions={routeLatLngs} />
                )}

                {/* Fit automatique sur l’itinéraire */}
                {routeLatLngs.length > 0 && (
                  <FitBounds
                    bounds={L.latLngBounds(
                      routeLatLngs.map(([lat, lng]) => [lat, lng])
                    )}
                  />
                )}
              </MapContainer>
            </div>
          </div>
        </div>

        {/* Activité récente — inchangé */}
        <div className="activityContainer">
          <h2>Activité récente</h2>
          <div className="activityColumns">
            {/* À venir */}
            <div className="upcomingBookings">
              <h3>Réservations à venir</h3>
              {upcomingBookings.length > 0 ? (
                (() => {
                  const nearest = [...upcomingBookings].sort(
                    (a, b) =>
                      new Date(a.scheduled_time) - new Date(b.scheduled_time)
                  )[0];
                  return (
                    <div className="bookingCard">
                      <p>
                        <strong>Destination :</strong>{" "}
                        {nearest.dropoff_location}
                      </p>
                      <p>
                        <strong>Date et Heure :</strong>{" "}
                        {new Date(nearest.scheduled_time).toLocaleString()}
                      </p>
                      <p>
                        <strong>Prix :</strong> {nearest.amount} €
                      </p>
                      <button
                        onClick={() =>
                          navigate(`/dashboard/booking-details/${nearest.id}`)
                        }
                      >
                        Consultez les détails
                      </button>
                    </div>
                  );
                })()
              ) : (
                <p>Aucune réservation à venir.</p>
              )}
            </div>

            {/* En cours (exemple conservé) */}
            <div className="ongoingBookings">
              <h3>Réservations en cours</h3>
              {ongoingBookings.length > 0 ? (
                (() => {
                  const now = new Date();
                  const today = now.toISOString().split("T")[0];
                  const valid = ongoingBookings
                    .filter((b) => {
                      const d = new Date(b.scheduled_time);
                      const day = d.toISOString().split("T")[0];
                      return day === today && d >= now;
                    })
                    .sort(
                      (a, b) =>
                        new Date(a.scheduled_time) - new Date(b.scheduled_time)
                    );
                  return valid.length ? (
                    <div className="bookingCard">
                      <p>
                        <strong>Destination :</strong>{" "}
                        {valid[0].dropoff_location}
                      </p>
                      <p>
                        <strong>Date et Heure :</strong>{" "}
                        {new Date(valid[0].scheduled_time).toLocaleString(
                          "fr-FR",
                          {
                            weekday: "long",
                            day: "2-digit",
                            month: "long",
                            year: "numeric",
                            hour: "2-digit",
                            minute: "2-digit",
                          }
                        )}
                      </p>
                      <p>
                        <strong>Prix :</strong> {valid[0].amount} €
                      </p>
                      <button
                        onClick={() =>
                          navigate(`/dashboard/booking-details/${valid[0].id}`)
                        }
                      >
                        Consultez les détails
                      </button>
                    </div>
                  ) : (
                    <p>Aucune réservation en cours aujourd'hui.</p>
                  );
                })()
              ) : (
                <p>Aucune réservation en cours.</p>
              )}
            </div>

            {/* Passées */}
            <div className="pastBookings">
              <h3>Réservations passées</h3>
              {pastBookings.length > 0 ? (
                (() => {
                  const latest = [...pastBookings].sort(
                    (a, b) =>
                      new Date(b.scheduled_time) - new Date(a.scheduled_time)
                  )[0];
                  return (
                    <div className="bookingCard">
                      <p>
                        <strong>Destination :</strong> {latest.dropoff_location}
                      </p>
                      <p>
                        <strong>Date et Heure :</strong>{" "}
                        {new Date(latest.scheduled_time).toLocaleString()}
                      </p>
                      <p>
                        <strong>Prix :</strong> {latest.amount} €
                      </p>
                      <button
                        onClick={() =>
                          navigate(`/dashboard/booking-details/${latest.id}`)
                        }
                      >
                        Consultez les détails
                      </button>
                    </div>
                  );
                })()
              ) : (
                <p>Aucune réservation passée.</p>
              )}
            </div>
          </div>
        </div>

        {error && <p className="error">{error}</p>}
      </main>

      <Footer />
    </div>
  );
};

export default ClientDashboard;
