// frontend/src/pages/company/Settings/tabs/OperationsTab.jsx
import React, { useState, useEffect } from "react";
import styles from "../CompanySettings.module.css";
import ToggleField from "../../../../components/ui/ToggleField";
import {
  fetchOperationalSettings,
  updateOperationalSettings,
} from "../../../../services/settingsService";

const OperationsTab = () => {
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  const [form, setForm] = useState({
    service_area: "",
    max_daily_bookings: 50,
    dispatch_enabled: false,
    latitude: null,
    longitude: null,
  });

  // Charger les données
  useEffect(() => {
    const loadData = async () => {
      try {
        const data = await fetchOperationalSettings();
        setForm({
          service_area: data.service_area || "",
          max_daily_bookings: data.max_daily_bookings || 50,
          dispatch_enabled: data.dispatch_enabled || false,
          latitude: data.latitude || null,
          longitude: data.longitude || null,
        });
      } catch (err) {
        console.error("Failed to load operational settings:", err);
        setError("Impossible de charger les paramètres.");
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, []);

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setForm((prev) => ({
      ...prev,
      [name]: type === "checkbox" ? checked : value,
    }));
  };

  const handleToggle = (e) => {
    const { name, checked } = e.target;
    setForm((prev) => ({
      ...prev,
      [name]: checked,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setMessage("");
    setError("");
    setSaving(true);

    try {
      const payload = {
        service_area: form.service_area || null,
        max_daily_bookings: parseInt(form.max_daily_bookings) || 50,
        dispatch_enabled: form.dispatch_enabled,
        latitude: form.latitude ? parseFloat(form.latitude) : null,
        longitude: form.longitude ? parseFloat(form.longitude) : null,
      };

      await updateOperationalSettings(payload);
      setMessage("✅ Paramètres opérationnels enregistrés avec succès.");
    } catch (err) {
      console.error("Failed to update operational settings:", err);
      setError(
        err?.response?.data?.error ||
          err?.message ||
          "Erreur lors de la sauvegarde."
      );
    } finally {
      setSaving(false);
    }
  };

  const detectGPS = () => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          setForm((prev) => ({
            ...prev,
            latitude: position.coords.latitude.toFixed(6),
            longitude: position.coords.longitude.toFixed(6),
          }));
          setMessage("📍 Position détectée automatiquement.");
        },
        (err) => {
          setError("Impossible de détecter la position GPS.");
          console.error("GPS error:", err);
        }
      );
    } else {
      setError("Votre navigateur ne supporte pas la géolocalisation.");
    }
  };

  if (loading) {
    return (
      <div className={styles.loadingContainer}>
        <div className={styles.spinner}></div>
        <p>Chargement…</p>
      </div>
    );
  }

  return (
    <form className={styles.settingsForm} onSubmit={handleSubmit}>
      {message && <div className={styles.success}>{message}</div>}
      {error && <div className={styles.error}>{error}</div>}

      {/* Configuration opérationnelle */}
      <section className={styles.section}>
        <h2>🚗 Configuration opérationnelle</h2>

        <div className={styles.formGroup}>
          <label htmlFor="service_area">Zone de service</label>
          <input
            id="service_area"
            name="service_area"
            value={form.service_area}
            onChange={handleChange}
            placeholder="Genève, Vaud, Valais"
          />
          <small className={styles.hint}>
            Zones géographiques couvertes (séparées par virgule)
          </small>
        </div>

        <div className={styles.formGroup}>
          <label htmlFor="max_daily_bookings">Limite de courses par jour</label>
          <input
            type="number"
            id="max_daily_bookings"
            name="max_daily_bookings"
            value={form.max_daily_bookings}
            onChange={handleChange}
            min="1"
            max="500"
          />
          <small className={styles.hint}>
            Nombre maximum de réservations acceptées quotidiennement
          </small>
        </div>
      </section>

      {/* Dispatch automatique */}
      <section className={styles.section}>
        <h2>🤖 Dispatch automatique</h2>

        <ToggleField
          label="Activer le dispatch automatique"
          name="dispatch_enabled"
          value={form.dispatch_enabled}
          onChange={handleToggle}
          hint={
            form.dispatch_enabled
              ? "✅ Le système planifie automatiquement les courses"
              : "⚠️ Vous devez assigner manuellement les chauffeurs"
          }
        />
      </section>

      {/* Géolocalisation */}
      <section className={styles.section}>
        <h2>📍 Géolocalisation</h2>

        <div className={styles.gpsRow}>
          <div className={styles.formGroup}>
            <label htmlFor="latitude">Latitude</label>
            <input
              type="number"
              id="latitude"
              name="latitude"
              value={form.latitude || ""}
              onChange={handleChange}
              step="0.000001"
              placeholder="46.2044"
            />
          </div>

          <div className={styles.formGroup}>
            <label htmlFor="longitude">Longitude</label>
            <input
              type="number"
              id="longitude"
              name="longitude"
              value={form.longitude || ""}
              onChange={handleChange}
              step="0.000001"
              placeholder="6.1432"
            />
          </div>

          <button
            type="button"
            className={`${styles.button} ${styles.secondary}`}
            onClick={detectGPS}
          >
            📍 Détecter
          </button>
        </div>

        <small className={styles.hint}>
          Coordonnées du siège social, utilisées pour les calculs de distance
        </small>
      </section>

      {/* Boutons */}
      <div className={styles.actionsRow}>
        <button
          type="submit"
          className={`${styles.button} ${styles.primary}`}
          disabled={saving}
        >
          {saving ? "💾 Enregistrement…" : "💾 Enregistrer"}
        </button>
      </div>
    </form>
  );
};

export default OperationsTab;
