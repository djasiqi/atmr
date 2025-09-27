// src/pages/company/Dashboard/components/DriverLiveMap.jsx
import React, { useEffect, useRef, useState } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { getCompanySocket } from "../../../../services/companySocket";
import useCompanyData from "../../../../hooks/useCompanyData";

// Icône Leaflet par défaut (corrige le bug d'icône manquante)
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
  iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
  shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
});

const defaultCenter = [46.8182, 8.2275]; // CH

// ---- helpers coords -------------------------------------------------
const toNumOrNull = (v) => {
  if (v === null || v === undefined) return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
};
const toLatLngSafe = (lat, lon) => {
  const la = toNumOrNull(lat);
  const lo = toNumOrNull(lon);
  return la !== null && lo !== null ? [la, lo] : null;
};
const resolveDriverCoords = (d) =>
  toLatLngSafe(d.current_lat, d.current_lon) ||
  toLatLngSafe(d.latitude, d.longitude) ||
  toLatLngSafe(d.last_latitude, d.last_longitude) ||
  (d.last_position && toLatLngSafe(d.last_position.lat, d.last_position.lon)) ||
  null;

export default function DriverLiveMap() {
  const mapRef = useRef(null);
  const mapElRef = useRef(null);
  const markersRef = useRef({}); // { [driverId]: L.Marker }
  const [driverLocations, setDriverLocations] = useState({});
  const { driver: staticDrivers, company } = useCompanyData();

  // petits helpers pour éviter d'appeler Leaflet sur une map détruite
  const getMap = () => {
    const m = mapRef.current;
    // _mapPane est défini une fois la map initialisée
    if (!m || !m._mapPane) return null;
    return m;
  };
  const safeSetView = (center, zoom) => {
    const m = getMap();
    if (!m) return;
    try { m.setView(center, zoom, { animate: false }); } catch {}
  };
  const fitBoundsToMarkers = (maxZoom = 14) => {
    const m = getMap();
    if (!m) return;
    const entries = Object.values(markersRef.current);
    if (entries.length === 0) return;
    try {
      const group = L.featureGroup(entries);
      m.fitBounds(group.getBounds().pad(0.2));
      if (m.getZoom() > maxZoom) m.setZoom(maxZoom);
    } catch {}
  };

  // Init carte Leaflet
  useEffect(() => {
    if (mapRef.current) return; // évite double init hors StrictMode
    if (!mapElRef.current) return;

    const map = L.map(mapElRef.current, {
      center: defaultCenter,
      zoom: 9,
    });
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution:
        '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> contributors',
    }).addTo(map);

    mapRef.current = map;

    return () => {
      // ⚠️ StrictMode va appeler le cleanup immédiatement en dev -> remets tout à zéro
      try { map.remove(); } catch {}
      mapRef.current = null;
      markersRef.current = {};
      setDriverLocations({});
    };
  }, []);

  // Placer les positions statiques au chargement
  useEffect(() => {
    const map = getMap();
    if (!map || !Array.isArray(staticDrivers)) return;

    let placed = 0;
    staticDrivers.forEach((d) => {
      if (markersRef.current[d.id]) return; // déjà placé (live)
      const ll = resolveDriverCoords(d);
      if (!ll) return; // ignore si pas de coords valides

      const label = d.first_name || d.username || d.name || `Driver ${d.id}`;
      const m = L.marker(ll).addTo(map);
      m.bindTooltip(label, {
        permanent: true,
        direction: "top",
        offset: [0, -24],
        className: "live-driver-label",
      }).openTooltip();

      markersRef.current[d.id] = m;
      placed++;
    });

    if (placed > 0) fitBoundsToMarkers();
    else if (Object.keys(markersRef.current).length === 0) {
      // aucun marker : vue par défaut
      safeSetView(defaultCenter, 9);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [staticDrivers]);

  // Socket: écouter les mises à jour live
  useEffect(() => {
    const socket = getCompanySocket();
    if (!socket) return;

    if (company?.id) {
      try { socket.emit("join_company", { company_id: company.id }); } catch {}
    }

    const onLoc = (data) => {
      const map = getMap();
      if (!map) return;

      // Backend : { driver_id, lat|latitude, lon|lng|longitude, ... }
      const id = data.driver_id ?? data.id;
      const lat = data.lat ?? data.latitude ?? data.current_lat;
      const lon = data.lon ?? data.lng ?? data.longitude ?? data.current_lon;
      const ll = toLatLngSafe(lat, lon);
      if (!id || !ll) return;

      const firstName = data.first_name || data.name || `Driver ${id}`;

      setDriverLocations((prev) => ({
        ...prev,
        [id]: { lat: ll[0], lon: ll[1], name: firstName },
      }));

      if (!markersRef.current[id]) {
        const m = L.marker(ll).addTo(map);
        m.bindTooltip(firstName, {
          permanent: true,
          direction: "top",
          offset: [0, -24],
          className: "live-driver-label",
        }).openTooltip();
        markersRef.current[id] = m;
      } else {
        markersRef.current[id].setLatLng(ll);
        const tt = markersRef.current[id].getTooltip();
        if (tt) tt.setContent(firstName);
      }

      fitBoundsToMarkers(14);
    };

    socket.on("driver_location", onLoc);
    socket.on("driver_location_update", onLoc);
    
    // Explicitly request driver locations when component mounts
    try {
      socket.emit("get_driver_locations");
    } catch (e) {
      console.error("Failed to request driver locations:", e);
    }

    return () => {
      socket.off("driver_location", onLoc);
      socket.off("driver_location_update", onLoc);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [company?.id]);

  return (
    <div style={{ width: "100%", height: 400, position: "relative" }}>
      <div ref={mapElRef} style={{ width: "100%", height: "100%" }} />
      <div
        style={{
          position: "absolute",
          right: 12,
          bottom: 12,
          background: "rgba(255,255,255,0.9)",
          border: "1px solid #ddd",
          borderRadius: 8,
          padding: "6px 10px",
          fontSize: 12,
        }}
      >
        {Object.keys(driverLocations).length} chauffeur(s) en direct
      </div>
    </div>
  );
}
