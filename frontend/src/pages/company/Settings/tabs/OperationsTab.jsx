// frontend/src/pages/company/Settings/tabs/OperationsTab.jsx
import React, { useState, useEffect, useCallback } from 'react';
import { FiTruck, FiMapPin, FiSettings, FiRefreshCw, FiEdit, FiX } from 'react-icons/fi';
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

const hasCompanyToken = () =>
  !!(localStorage.getItem('company_access_token') || localStorage.getItem('company_authToken'));

const OperationsTab = ({ isEditing: _isEditing }) => {
  const [loading, setLoading] = useState(true);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [currentMode, setCurrentMode] = useState('semi_auto');
  const [showAdvancedSettingsModal, setShowAdvancedSettingsModal] = useState(false);
  const [advancedSettings, setAdvancedSettings] = useState(null);
  const [loadingAdvancedSettings, setLoadingAdvancedSettings] = useState(false);
  const [drivers, setDrivers] = useState([]);

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
  const loadAdvancedSettings = useCallback(async () => {
    if (!hasCompanyToken()) return;
    setLoadingAdvancedSettings(true);
    try {
      const { data } = await apiClient.get('/company_dispatch/advanced_settings');
      setAdvancedSettings(data.dispatch_overrides);
    } catch (err) {
      console.error('[OperationsTab] Erreur chargement parametres avances:', err);
    } finally {
      setLoadingAdvancedSettings(false);
    }
  }, []);

  // Sauvegarder les paramètres avancés
  const saveAdvancedSettings = async (newSettings) => {
    try {
      const { data } = await apiClient.put('/company_dispatch/advanced_settings', {
        dispatch_overrides: newSettings,
      });
      setAdvancedSettings(data.dispatch_overrides);
      setShowAdvancedSettingsModal(false);
      showSuccess('Paramètres avancés sauvegardés avec succès.');
      console.log('[OperationsTab] Paramètres avancés sauvegardés:', data.dispatch_overrides);
      await loadAdvancedSettings();
    } catch (err) {
      console.error('[OperationsTab] Erreur sauvegarde paramètres avancés:', err);
      showError('Erreur lors de la sauvegarde des paramètres');
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
      showSuccess('Paramètres réinitialisés aux valeurs par défaut.');
      console.log('[OperationsTab] Paramètres avancés réinitialisés');
    } catch (err) {
      console.error('[OperationsTab] Erreur réinitialisation paramètres:', err);
      showError('Erreur lors de la réinitialisation');
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

        // Charger aussi le mode de dispatch actuel (seulement si token dispo)
        if (hasCompanyToken()) {
          try {
            const { data: modeData } = await apiClient.get('/company_dispatch/mode');
            if (modeData.dispatch_mode) {
              setCurrentMode(modeData.dispatch_mode);
            }
          } catch (err) {
            console.error('Failed to load dispatch mode:', err);
          }
        }
      } catch (err) {
        console.error('Failed to load operational settings:', err);
        setError('Impossible de charger les parametres.');
      } finally {
        setLoading(false);
      }
    };

    loadData();
    loadAdvancedSettings();

    // Charger les chauffeurs pour la sélection de préférence
    const loadDrivers = async () => {
      try {
        const { data } = await apiClient.get('/companies/me/drivers');
        // Normaliser la réponse (peut être un tableau ou un objet avec drivers)
        const driversList = Array.isArray(data) ? data : data?.drivers || [];
        setDrivers(driversList);
      } catch (err) {
        console.error('[OperationsTab] Erreur chargement chauffeurs:', err);
      }
    };
    loadDrivers();
  }, [loadAdvancedSettings]);

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
      setMessage('Sauvegardé automatiquement.');
      setTimeout(() => setMessage(''), 2000);
    } catch (err) {
      console.error('Auto-save failed:', err);
      setError('Erreur lors de la sauvegarde');
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
            setMessage('Position détectée et sauvegardée automatiquement.');
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
    <div className={`${styles.settingsForm} ${styles.blockDisplay}`}>
      {message && <div className={styles.success}>{message}</div>}
      {error && <div className={styles.error}>{error}</div>}

      <div className={styles.operationsGrid}>
        <div className={styles.operationsCol}>
          {/* Carte 1 : Configuration operationnelle */}
          <div className={styles.card}>
            <div className={styles.cardHeader}>
              <div className={styles.cardIcon}><FiTruck size={16} /></div>
              <div className={styles.cardHeaderText}>
                <h3 className={styles.cardTitle}>Configuration operationnelle</h3>
                <p className={styles.cardHint}>Zones couvertes et limites de capacite</p>
              </div>
            </div>

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
          </div>

          {/* Carte 2 : Geolocalisation */}
          <div className={styles.card}>
            <div className={styles.cardHeader}>
              <div className={styles.cardIcon}><FiMapPin size={16} /></div>
              <div className={styles.cardHeaderText}>
                <h3 className={styles.cardTitle}>Geolocalisation</h3>
                <p className={styles.cardHint}>Coordonnees du siege social pour les calculs de distance</p>
              </div>
            </div>

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
                <FiMapPin aria-hidden /> Détecter
              </button>
            </div>
          </div>

          {/* Carte 3 : Configuration Dispatch Avancee (masquee en mode manuel) */}
          {currentMode !== 'manual' && <div className={styles.card}>
            <div className={styles.cardHeader}>
              <div className={styles.cardIcon}><FiSettings size={16} /></div>
              <div className={styles.cardHeaderText}>
                <h3 className={styles.cardTitle}>Configuration Dispatch Avancee</h3>
                <p className={styles.cardHint}>Heuristiques, solver, equite, chauffeurs d'urgence</p>
              </div>
            </div>

            <div className={styles.advancedActionsRow}>
              <button
                type="button"
                className={`${styles.button} ${styles.primary}`}
                onClick={() => setShowAdvancedSettingsModal(true)}
                disabled={loadingAdvancedSettings}
              >
                {advancedSettings ? (
                  <>
                    <FiEdit aria-hidden /> Modifier les paramètres
                  </>
                ) : (
                  <>
                    <FiSettings aria-hidden /> Configurer
                  </>
                )}
              </button>

              {advancedSettings && (
                <button
                  type="button"
                  className={`${styles.button} ${styles.secondary}`}
                  onClick={resetAdvancedSettings}
                  disabled={loadingAdvancedSettings}
                >
                  <FiRefreshCw aria-hidden /> Réinitialiser
                </button>
              )}

              {loadingAdvancedSettings && (
                <span className={styles.statusText}>Chargement...</span>
              )}

              {advancedSettings && !loadingAdvancedSettings && (
                <span className={styles.statusTextSuccess}>Paramètres personnalisés actifs</span>
              )}
            </div>

            {!advancedSettings && !loadingAdvancedSettings && (
              <p className={styles.advancedHint}>
                Aucune configuration personnalisée. Les valeurs par défaut seront utilisées.
              </p>
            )}
          </div>}
        </div>

        <div>
          {/* Carte 4 : Systeme de dispatch autonome */}
          <div className={styles.card}>
            <DispatchModeSelector onModeChange={handleModeChange} />
            <AutonomousConfigPanel visible={currentMode === 'fully_auto'} />
          </div>
        </div>
      </div>

      {showAdvancedSettingsModal && (
        <div
          className={styles.modalOverlay}
          onClick={() => setShowAdvancedSettingsModal(false)}
          role="presentation"
        >
          <div
            className={styles.modalContentLarge}
            onClick={(e) => e.stopPropagation()}
            role="dialog"
            aria-modal="true"
          >
            <button
              type="button"
              className={styles.modalClose}
              onClick={() => setShowAdvancedSettingsModal(false)}
              aria-label="Fermer"
            >
              <FiX aria-hidden />
            </button>
            <AdvancedSettings
              onApply={saveAdvancedSettings}
              initialSettings={advancedSettings || {}}
              drivers={drivers}
            />
          </div>
        </div>
      )}
    </div>
  );
};

export default OperationsTab;
