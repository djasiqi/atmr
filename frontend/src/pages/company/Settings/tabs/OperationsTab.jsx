// frontend/src/pages/company/Settings/tabs/OperationsTab.jsx
import React, { useState, useEffect } from 'react';
import styles from '../CompanySettings.module.css';
import DispatchModeSelector from '../../../../components/DispatchModeSelector';
import AutonomousConfigPanel from '../../../../components/AutonomousConfigPanel';
import AdvancedSettings from '../../Dispatch/components/AdvancedSettings';
import {
  fetchOperationalSettings,
  updateOperationalSettings,
} from '../../../../services/settingsService';
import apiClient from '../../../../utils/apiClient';
import { showSuccess, showError } from '../../../../utils/toast';

const OperationsTab = () => {
  const [loading, setLoading] = useState(true);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [currentMode, setCurrentMode] = useState('semi_auto');
  const [showAdvancedSettingsModal, setShowAdvancedSettingsModal] = useState(false);
  const [advancedSettings, setAdvancedSettings] = useState(null);
  const [loadingAdvancedSettings, setLoadingAdvancedSettings] = useState(false);

  const [form, setForm] = useState({
    service_area: '',
    max_daily_bookings: 50,
    dispatch_enabled: false,
    latitude: null,
    longitude: null,
  });

  const handleModeChange = (newMode) => {
    setCurrentMode(newMode);
  };

  // Charger les paramètres avancés depuis la DB
  const loadAdvancedSettings = async () => {
    setLoadingAdvancedSettings(true);
    try {
      const { data } = await apiClient.get('/company_dispatch/advanced_settings');
      setAdvancedSettings(data.dispatch_overrides);
      console.log('🔄 [OperationsTab] Paramètres avancés chargés:', data.dispatch_overrides);
    } catch (err) {
      console.error('[OperationsTab] Erreur chargement paramètres avancés:', err);
    } finally {
      setLoadingAdvancedSettings(false);
    }
  };

  // Sauvegarder les paramètres avancés
  const saveAdvancedSettings = async (newSettings) => {
    try {
      const { data } = await apiClient.put('/company_dispatch/advanced_settings', {
        dispatch_overrides: newSettings,
      });
      setAdvancedSettings(data.dispatch_overrides);
      setShowAdvancedSettingsModal(false);
      showSuccess('✅ Paramètres avancés sauvegardés avec succès !');
      console.log('💾 [OperationsTab] Paramètres avancés sauvegardés:', data.dispatch_overrides);
    } catch (err) {
      console.error('[OperationsTab] Erreur sauvegarde paramètres avancés:', err);
      showError('❌ Erreur lors de la sauvegarde des paramètres');
    }
  };

  // Réinitialiser les paramètres avancés
  const resetAdvancedSettings = async () => {
    if (!window.confirm('Réinitialiser tous les paramètres avancés aux valeurs par défaut ?')) {
      return;
    }

    try {
      await apiClient.delete('/company_dispatch/advanced_settings');
      setAdvancedSettings(null);
      showSuccess('✅ Paramètres réinitialisés aux valeurs par défaut');
      console.log('🗑️ [OperationsTab] Paramètres avancés réinitialisés');
    } catch (err) {
      console.error('[OperationsTab] Erreur réinitialisation paramètres:', err);
      showError('❌ Erreur lors de la réinitialisation');
    }
  };

  // Charger les données
  useEffect(() => {
    const loadData = async () => {
      try {
        const data = await fetchOperationalSettings();
        setForm({
          service_area: data.service_area || '',
          max_daily_bookings: data.max_daily_bookings || 50,
          dispatch_enabled: data.dispatch_enabled || false,
          latitude: data.latitude || null,
          longitude: data.longitude || null,
        });

        // Charger aussi le mode de dispatch actuel
        try {
          const apiClient = (await import('../../../../utils/apiClient')).default;
          const { data: modeData } = await apiClient.get('/company_dispatch/mode');
          if (modeData.dispatch_mode) {
            setCurrentMode(modeData.dispatch_mode);
          }
        } catch (err) {
          console.error('Failed to load dispatch mode:', err);
        }
      } catch (err) {
        console.error('Failed to load operational settings:', err);
        setError('Impossible de charger les paramètres.');
      } finally {
        setLoading(false);
      }
    };

    loadData();
    loadAdvancedSettings(); // Charger aussi les paramètres avancés
  }, []);

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setForm((prev) => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value,
    }));
  };

  // Sauvegarde automatique quand l'utilisateur quitte un champ
  const autoSave = async (fieldName, fieldValue) => {
    setMessage('');
    setError('');

    try {
      // Construire le payload avec les bonnes valeurs
      const latitudeValue = fieldName === 'latitude' ? fieldValue : form.latitude;
      const longitudeValue = fieldName === 'longitude' ? fieldValue : form.longitude;

      const payload = {
        max_daily_bookings:
          fieldName === 'max_daily_bookings'
            ? parseInt(fieldValue) || 50
            : parseInt(form.max_daily_bookings) || 50,
        dispatch_enabled: form.dispatch_enabled || false,
      };

      // Ajouter les champs optionnels seulement s'ils ne sont pas null
      const serviceAreaValue = fieldName === 'service_area' ? fieldValue : form.service_area;
      if (serviceAreaValue && serviceAreaValue !== '') {
        payload.service_area = serviceAreaValue;
      }

      if (latitudeValue && latitudeValue !== '') {
        payload.latitude = parseFloat(latitudeValue);
      }

      if (longitudeValue && longitudeValue !== '') {
        payload.longitude = parseFloat(longitudeValue);
      }

      await updateOperationalSettings(payload);
      setMessage('✅ Sauvegardé automatiquement');
      setTimeout(() => setMessage(''), 2000);
    } catch (err) {
      console.error('Auto-save failed:', err);
      setError('❌ Erreur lors de la sauvegarde');
      setTimeout(() => setError(''), 3000);
    }
  };

  const handleBlur = (e) => {
    const { name, value } = e.target;
    autoSave(name, value);
  };

  const detectGPS = () => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        async (position) => {
          const newLat = position.coords.latitude.toFixed(6);
          const newLng = position.coords.longitude.toFixed(6);

          setForm((prev) => ({
            ...prev,
            latitude: newLat,
            longitude: newLng,
          }));

          // Sauvegarder automatiquement après détection
          try {
            const gpsPayload = {
              max_daily_bookings: parseInt(form.max_daily_bookings) || 50,
              dispatch_enabled: form.dispatch_enabled || false,
              latitude: parseFloat(newLat),
              longitude: parseFloat(newLng),
            };

            // Ajouter service_area seulement si non vide
            if (form.service_area && form.service_area !== '') {
              gpsPayload.service_area = form.service_area;
            }

            await updateOperationalSettings(gpsPayload);
            setMessage('📍 Position détectée et sauvegardée automatiquement');
            setTimeout(() => setMessage(''), 2000);
          } catch (err) {
            console.error('Failed to save GPS:', err);
            setError('Position détectée mais échec de la sauvegarde');
            setTimeout(() => setError(''), 3000);
          }
        },
        (err) => {
          setError('Impossible de détecter la position GPS.');
          console.error('GPS error:', err);
          setTimeout(() => setError(''), 3000);
        }
      );
    } else {
      setError('Votre navigateur ne supporte pas la géolocalisation.');
      setTimeout(() => setError(''), 3000);
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
    <div className={styles.settingsForm} style={{ display: 'block' }}>
      {message && <div className={styles.success}>{message}</div>}
      {error && <div className={styles.error}>{error}</div>}

      {/* Layout en 2 colonnes */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: 'var(--spacing-lg)',
          alignItems: 'start',
          width: '100%',
        }}
      >
        {/* COLONNE GAUCHE */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--spacing-md)' }}>
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
                onBlur={handleBlur}
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
                onBlur={handleBlur}
                min="1"
                max="500"
              />
              <small className={styles.hint}>
                Nombre maximum de réservations acceptées quotidiennement
              </small>
            </div>
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
                  value={form.latitude || ''}
                  onChange={handleChange}
                  onBlur={handleBlur}
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
                  value={form.longitude || ''}
                  onChange={handleChange}
                  onBlur={handleBlur}
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

          {/* 🆕 Configuration Dispatch Avancée */}
          <section className={styles.section}>
            <h2>⚙️ Configuration Dispatch Avancée</h2>
            <p style={{ color: 'var(--text-secondary)', marginBottom: 'var(--spacing-md)' }}>
              Personnalisez finement les paramètres de dispatch (heuristiques, solver, équité,
              chauffeurs d'urgence, etc.)
            </p>

            <div
              style={{
                display: 'flex',
                gap: 'var(--spacing-sm)',
                alignItems: 'center',
                flexWrap: 'wrap',
              }}
            >
              <button
                type="button"
                className={`${styles.button} ${styles.primary}`}
                onClick={() => setShowAdvancedSettingsModal(true)}
                disabled={loadingAdvancedSettings}
              >
                {advancedSettings ? '✏️ Modifier les paramètres' : '⚙️ Configurer'}
              </button>

              {advancedSettings && (
                <button
                  type="button"
                  className={`${styles.button} ${styles.secondary}`}
                  onClick={resetAdvancedSettings}
                  disabled={loadingAdvancedSettings}
                >
                  🔄 Réinitialiser
                </button>
              )}

              {loadingAdvancedSettings && (
                <span style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
                  Chargement...
                </span>
              )}

              {advancedSettings && !loadingAdvancedSettings && (
                <span style={{ color: 'var(--success)', fontSize: '0.9rem' }}>
                  ✅ Paramètres personnalisés actifs
                </span>
              )}
            </div>

            {!advancedSettings && !loadingAdvancedSettings && (
              <p
                style={{
                  color: 'var(--text-muted)',
                  fontSize: '0.85rem',
                  marginTop: 'var(--spacing-sm)',
                }}
              >
                💡 Aucune configuration personnalisée. Les valeurs par défaut seront utilisées.
              </p>
            )}
          </section>
        </div>

        {/* COLONNE DROITE */}
        <div>
          {/* Système de dispatch autonome */}
          <section className={styles.section}>
            <DispatchModeSelector onModeChange={handleModeChange} />
            <AutonomousConfigPanel visible={currentMode === 'fully_auto'} />
          </section>
        </div>
      </div>

      {/* Modal Paramètres Avancés */}
      {showAdvancedSettingsModal && (
        <div
          className="modal-overlay"
          onClick={() => setShowAdvancedSettingsModal(false)}
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.5)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 9999,
          }}
        >
          <div
            className="modal-content"
            onClick={(e) => e.stopPropagation()}
            style={{
              backgroundColor: 'white',
              borderRadius: '8px',
              maxWidth: '800px',
              maxHeight: '90vh',
              overflow: 'auto',
              position: 'relative',
              padding: '20px',
              boxShadow: '0 4px 20px rgba(0, 0, 0, 0.2)',
            }}
          >
            <button
              onClick={() => setShowAdvancedSettingsModal(false)}
              style={{
                position: 'absolute',
                top: '10px',
                right: '10px',
                background: 'transparent',
                border: 'none',
                fontSize: '1.5rem',
                cursor: 'pointer',
                color: 'var(--text-secondary)',
              }}
            >
              ✕
            </button>
            <AdvancedSettings
              onApply={saveAdvancedSettings}
              initialSettings={advancedSettings || {}}
            />
          </div>
        </div>
      )}
    </div>
  );
};

export default OperationsTab;
