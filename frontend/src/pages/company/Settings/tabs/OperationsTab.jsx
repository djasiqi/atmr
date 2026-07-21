// frontend/src/pages/company/Settings/tabs/OperationsTab.jsx
import React, { forwardRef, useState, useEffect, useCallback, useRef, useImperativeHandle } from 'react';
import { FiTruck, FiMapPin, FiSettings, FiRefreshCw, FiEdit, FiX } from 'react-icons/fi';
import styles from '../CompanySettings.module.css';
import DispatchModeSelector from '../../../../components/DispatchModeSelector';
import AutonomousConfigPanel from '../../../../components/AutonomousConfigPanel';
import AdvancedSettings from '../../Dispatch/components/AdvancedSettings';
import ServiceAreaZonesAutocomplete from '../../../../components/common/ServiceAreaZonesAutocomplete';
import {
  fetchOperationalSettings,
  fetchServiceAreaZones,
  updateOperationalSettings,
} from '../../../../services/settingsService';
import apiClient from '../../../../utils/apiClient';
import { showSuccess, showError } from '../../../../utils/toast';
import {
  getAuthEnv,
  hasCompanyScopedAccessToken,
} from '../../../../utils/webAuthSession';

const hasCompanyToken = () =>
  hasCompanyScopedAccessToken(getAuthEnv());

const SERVICE_AREA_ALLOWED_TYPES = new Set(['commune', 'district', 'canton']);
const SERVICE_AREA_SINGLE_MODES = new Set(['canton', 'district']);
const SERVICE_AREA_JSON_VERSION = 1;
const SERVICE_AREA_TOKEN_REGEX = /^(commune|district|canton):[A-Za-z0-9_-]+$/;
const SERVICE_AREA_NAMED_REGEX = /^(commune_name|canton_name|district_name):.+$/;

const isPersistableServiceAreaToken = (token) =>
  SERVICE_AREA_TOKEN_REGEX.test(String(token || ''));

const resolveCanonicalServiceAreaToken = (item) => {
  const token = String(item?.token || '').trim();
  if (SERVICE_AREA_TOKEN_REGEX.test(token)) {
    return token;
  }
  const zoneType = String(item?.type || '').toLowerCase();
  const code = String(item?.code || '').trim();
  const cantonCode = String(item?.canton_code || '').trim();
  if (zoneType === 'canton' && /^[A-Za-z0-9_-]+$/.test(cantonCode)) {
    return `canton:${cantonCode}`;
  }
  if (zoneType === 'canton' && /^[A-Za-z0-9_-]+$/.test(code)) {
    return `canton:${code}`;
  }
  if (zoneType === 'district' && /^[A-Za-z0-9_-]+$/.test(code)) {
    return `district:${code}`;
  }
  if (zoneType === 'commune' && /^[A-Za-z0-9_-]+$/.test(code)) {
    return `commune:${code}`;
  }
  if (SERVICE_AREA_NAMED_REGEX.test(token)) {
    return null;
  }
  const nextId = Number(item?.id);
  if (Number.isFinite(nextId) && SERVICE_AREA_ALLOWED_TYPES.has(zoneType)) {
    return `${zoneType}:${nextId}`;
  }
  return null;
};
const TOKEN_LOOKS_RAW_REGEX = /^(commune|district|canton):[A-Za-z0-9_-]+$/;
const TYPE_LABELS = {
  commune: 'Commune',
  district: 'District',
  canton: 'Canton',
};

const inferModeFromToken = (token) => String(token || '').split(':')[0] || null;

const normalizeMode = (mode) => {
  const value = String(mode || '').trim().toLowerCase();
  return SERVICE_AREA_ALLOWED_TYPES.has(value) ? value : null;
};

const parseServiceAreaConfig = (rawValue) => {
  const raw = String(rawValue || '').trim();
  if (!raw) {
    return { mode: null, tokens: [], legacyValue: '' };
  }

  try {
    const parsed = JSON.parse(raw);
    const mode = normalizeMode(parsed?.mode);
    const tokens = Array.isArray(parsed?.tokens)
      ? parsed.tokens.map((token) => String(token).trim()).filter(Boolean)
      : [];
    const validTokens = tokens.filter(
      (token) => SERVICE_AREA_TOKEN_REGEX.test(token) || SERVICE_AREA_NAMED_REGEX.test(token)
    );
    const version = Number(parsed?.v);
    if (version === SERVICE_AREA_JSON_VERSION && mode && validTokens.length > 0) {
      return { mode, tokens: validTokens, legacyValue: '' };
    }
  } catch (_error) {
    // Fallback legacy (string CSV tokens)
  }

  const parts = raw
    .split(',')
    .map((part) => part.trim())
    .filter(Boolean);
  const validLegacyTokens = parts.filter(
    (token) => SERVICE_AREA_TOKEN_REGEX.test(token) || SERVICE_AREA_NAMED_REGEX.test(token)
  );
  if (validLegacyTokens.length > 0) {
    return { mode: inferModeFromToken(validLegacyTokens[0]), tokens: validLegacyTokens, legacyValue: '' };
  }
  return { mode: null, tokens: [], legacyValue: raw };
};

const serializeServiceAreaConfig = (mode, tokens) => {
  const normalizedMode = normalizeMode(mode);
  const normalizedTokens = Array.isArray(tokens)
    ? tokens.map((token) => String(token || '').trim()).filter(Boolean)
    : [];
  if (!normalizedMode || normalizedTokens.length === 0) {
    return '';
  }
  const payload = {
    v: SERVICE_AREA_JSON_VERSION,
    mode: normalizedMode,
    tokens: normalizedTokens,
  };
  return JSON.stringify(payload);
};

const buildZoneDisplayName = (zone) => {
  const rawName = String(zone?.name || '').trim();
  const token = String(zone?.token || '').trim();
  const zoneType = String(zone?.type || '').toLowerCase();
  const typeLabel = TYPE_LABELS[zoneType] || 'Zone';
  const canton = String(zone?.canton_code || '').trim();
  const rawCode = token.split(':')[1] || zone?.code || '';

  if (rawName && !TOKEN_LOOKS_RAW_REGEX.test(rawName)) {
    return rawName;
  }
  if (zoneType === 'canton' && rawCode) return `Canton ${rawCode}`;
  if (zoneType === 'district' && rawCode) return `District ${rawCode}`;
  if (zoneType === 'commune' && rawCode) {
    return canton ? `${typeLabel} ${rawCode} (${canton})` : `${typeLabel} ${rawCode}`;
  }
  return token || rawName || typeLabel;
};

const OperationsTab = forwardRef(({ isEditing }, ref) => {
  const [loading, setLoading] = useState(true);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [currentMode, setCurrentMode] = useState('semi_auto');
  const [showAdvancedSettingsModal, setShowAdvancedSettingsModal] = useState(false);
  const [advancedSettings, setAdvancedSettings] = useState(null);
  const [loadingAdvancedSettings, setLoadingAdvancedSettings] = useState(false);
  const [drivers, setDrivers] = useState([]);
  const [serviceAreaZones, setServiceAreaZones] = useState([]);
  const [serviceAreaMode, setServiceAreaMode] = useState(null);
  const [legacyServiceArea, setLegacyServiceArea] = useState('');
  const [allowFallbackResults, setAllowFallbackResults] = useState(false);
  const serverSnapshotRef = useRef({
    form: null,
    serviceAreaZones: [],
    serviceAreaMode: null,
    legacyServiceArea: '',
  });

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

  const hydrateOperationalData = useCallback(async (data) => {
    const nextForm = {
      service_area: data.service_area || '',
      max_daily_bookings: data.max_daily_bookings || 50,
      dispatch_enabled: data.dispatch_enabled || false,
      latitude: data.latitude || null,
      longitude: data.longitude || null,
    };
    setForm(nextForm);

    const parsed = parseServiceAreaConfig(data.service_area || '');
    let nextZones = [];
    let nextMode = null;
    let nextLegacy = '';
    if (parsed.tokens.length > 0) {
      const hydrated = await fetchServiceAreaZones({
        tokens: parsed.tokens,
        types: 'commune,canton,district',
        limit: 50,
      });
      const byToken = new Map(
        hydrated
          .filter((item) => item?.token)
          .map((item) => [String(item.token), item])
      );
      nextZones = parsed.tokens
        .map((token) => {
          const found = byToken.get(token);
          if (!found) return null;
          return {
            id: Number.isFinite(Number(found.id)) ? Number(found.id) : null,
            type: String(found.type),
            name: found.name,
            code: found.code || null,
            canton_code: found.canton_code || null,
            token: found.token || token,
            source: found.source || 'db',
            confidence: found.confidence || 'inferred',
          };
        })
        .filter(Boolean);
      nextMode = parsed.mode || normalizeMode(nextZones[0]?.type) || null;
      nextLegacy = '';
    } else {
      nextZones = [];
      nextMode = null;
      nextLegacy = parsed.legacyValue;
    }

    setServiceAreaZones(nextZones);
    setServiceAreaMode(nextMode);
    setLegacyServiceArea(nextLegacy);
    serverSnapshotRef.current = {
      form: nextForm,
      serviceAreaZones: nextZones,
      serviceAreaMode: nextMode,
      legacyServiceArea: nextLegacy,
    };
  }, []);

  // Charger les données
  useEffect(() => {
    const loadData = async () => {
      try {
        const data = await fetchOperationalSettings();
        await hydrateOperationalData(data);
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
  }, [hydrateOperationalData, loadAdvancedSettings]);

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setForm((prev) => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value,
    }));
  };

  const validateServiceAreaDraft = useCallback((mode, zones) => {
    if (!zones || zones.length === 0) {
      return 'Ajoute au moins une zone de service.';
    }
    if (!mode || !SERVICE_AREA_ALLOWED_TYPES.has(mode)) {
      return 'Mode de zone invalide.';
    }
    if (SERVICE_AREA_SINGLE_MODES.has(mode) && zones.length !== 1) {
      return `Le mode ${mode} exige une seule zone.`;
    }
    return null;
  }, []);

  const saveOperationalDraft = useCallback(async () => {
    setMessage('');
    setError('');

    try {
      const validationError = validateServiceAreaDraft(serviceAreaMode, serviceAreaZones);
      if (validationError) {
        setError(validationError);
        return;
      }
      const serialized = serializeServiceAreaConfig(
        serviceAreaMode,
        serviceAreaZones.map((zone) => zone.token)
      );

      const payload = {
        max_daily_bookings: parseInt(form.max_daily_bookings) || 50,
        dispatch_enabled: form.dispatch_enabled || false,
        service_area: serialized || '',
      };

      if (form.latitude && form.latitude !== '') {
        payload.latitude = parseFloat(form.latitude);
      }

      if (form.longitude && form.longitude !== '') {
        payload.longitude = parseFloat(form.longitude);
      }

      await updateOperationalSettings(payload);
      const refreshed = await fetchOperationalSettings();
      await hydrateOperationalData(refreshed);
      setMessage('Paramètres opérationnels enregistrés.');
      setTimeout(() => setMessage(''), 2500);
    } catch (err) {
      console.error('Save operational settings failed:', err);
      setError('Erreur lors de la sauvegarde');
      setTimeout(() => setError(''), 3000);
    }
  }, [form, serviceAreaMode, serviceAreaZones, hydrateOperationalData, validateServiceAreaDraft]);

  const resetOperationalDraft = useCallback(() => {
    const snapshot = serverSnapshotRef.current;
    if (snapshot?.form) {
      setForm(snapshot.form);
      setServiceAreaZones(snapshot.serviceAreaZones || []);
      setServiceAreaMode(snapshot.serviceAreaMode || null);
      setLegacyServiceArea(snapshot.legacyServiceArea || '');
    }
    setMessage('');
    setError('');
  }, []);

  useImperativeHandle(ref, () => ({
    save: saveOperationalDraft,
    reset: resetOperationalDraft,
    isReady: () => !loading,
  }), [saveOperationalDraft, resetOperationalDraft, loading]);

  const handleServiceAreaSelect = (item) => {
    if (!isEditing) return;
    const nextType = String(item?.type || '').toLowerCase();
    if (!SERVICE_AREA_ALLOWED_TYPES.has(nextType)) {
      showError('Type de zone non supporté.');
      return;
    }

    const nextIdRaw = item?.id;
    const nextId = Number.isFinite(Number(nextIdRaw)) ? Number(nextIdRaw) : null;
    const nextToken = resolveCanonicalServiceAreaToken(item);
    if (!nextToken) {
      showError('Zone de service invalide.');
      return;
    }
    if (!isPersistableServiceAreaToken(nextToken)) {
      showError('Cette zone est en fallback et ne peut pas être persistée. Choisis une zone officielle.');
      return;
    }
    const duplicate = serviceAreaZones.some((zone) => {
      if (zone.token === nextToken) return true;
      const zoneHasId = Number.isFinite(Number(zone.id));
      const nextHasId = Number.isFinite(Number(nextId));
      return zoneHasId && nextHasId && Number(zone.id) === Number(nextId) && zone.type === nextType;
    });
    if (duplicate) {
      return;
    }

    const nextZone = {
      id: nextId,
      type: nextType,
      name: item.name,
      code: item.code || null,
      canton_code: item.canton_code || null,
      token: nextToken,
      source: item.source || 'db',
      confidence: item.confidence || 'inferred',
    };

    let nextMode = serviceAreaMode;
    let updatedZones = [...serviceAreaZones];
    if (!nextMode) {
      nextMode = nextType;
    } else if (nextMode !== nextType) {
      nextMode = nextType;
      updatedZones = [];
      showSuccess(`Mode de zone basculé sur "${nextType}". La sélection précédente a été remplacée.`);
    }

    if (SERVICE_AREA_SINGLE_MODES.has(nextMode)) {
      updatedZones = [nextZone];
    } else {
      updatedZones = [...updatedZones, nextZone];
    }

    // Dédup finale par token.
    const seenTokens = new Set();
    updatedZones = updatedZones.filter((zone) => {
      if (!zone.token || seenTokens.has(zone.token)) return false;
      seenTokens.add(zone.token);
      return true;
    });

    setServiceAreaZones(updatedZones);
    setServiceAreaMode(nextMode);
    setLegacyServiceArea('');
    setForm((prev) => ({
      ...prev,
      service_area: serializeServiceAreaConfig(nextMode, updatedZones.map((zone) => zone.token)),
    }));
  };

  const removeServiceAreaZone = (indexToRemove) => {
    if (!isEditing) return;
    const updatedZones = serviceAreaZones.filter((_, index) => index !== indexToRemove);
    const nextMode = updatedZones.length > 0 ? serviceAreaMode : null;
    const serialized = serializeServiceAreaConfig(nextMode, updatedZones.map((zone) => zone.token));
    setServiceAreaZones(updatedZones);
    setServiceAreaMode(nextMode);
    setLegacyServiceArea('');
    setForm((prev) => ({ ...prev, service_area: serialized }));
  };

  const detectGPS = () => {
    if (!isEditing) return;
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

          setMessage('Position détectée (non sauvegardée). Clique sur Enregistrer.');
          setTimeout(() => setMessage(''), 2500);
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
              <ServiceAreaZonesAutocomplete
                inputId="service_area"
                onSelect={handleServiceAreaSelect}
                placeholder="Commune, ville ou canton (ex. Genève, GE, Lausanne)"
                disabled={!isEditing}
                allowFallbackResults={allowFallbackResults}
              />
              <label className={styles.hint} style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 6 }}>
                <input
                  type="checkbox"
                  checked={allowFallbackResults}
                  onChange={(event) => setAllowFallbackResults(event.target.checked)}
                  disabled={!isEditing}
                />
                Afficher résultats fallback (moins fiables)
              </label>
              {serviceAreaZones.length > 0 && (
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginTop: 8 }}>
                  {serviceAreaZones.map((zone, index) => (
                    <span key={`${zone.token}-${index}`} className={styles.chip} title={zone.token || ''}>
                      {buildZoneDisplayName(zone)}
                      {' '}
                      ({zone.type === 'canton' ? 'canton' : zone.type === 'district' ? 'district' : 'commune'}
                      {zone.canton_code ? `, ${zone.canton_code}` : ''})
                      <button
                        type="button"
                        onClick={() => removeServiceAreaZone(index)}
                        aria-label={`Retirer ${zone.name}`}
                        className={`${styles.button} ${styles.secondary}`}
                        style={{ padding: '2px 8px', minHeight: 'auto' }}
                        disabled={!isEditing}
                      >
                        Retirer
                      </button>
                    </span>
                  ))}
                </div>
              )}
              {legacyServiceArea && serviceAreaZones.length === 0 && (
                <small className={styles.hint}>
                  Valeur legacy détectée: {legacyServiceArea}. Sélectionne une zone pour migrer vers le format JSON V1.
                </small>
              )}
              <small className={styles.hint}>
                Mode actuel: {serviceAreaMode || 'aucun'}.
                {' '}
                Commune = multi, canton/district = sélection unique.
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
                disabled={!isEditing}
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
                  disabled={!isEditing}
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
                  disabled={!isEditing}
                  step="0.000001"
                  placeholder="6.1432"
                />
              </div>

              <button
                type="button"
                className={`${styles.button} ${styles.secondary}`}
                onClick={detectGPS}
                disabled={!isEditing}
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
});

export default OperationsTab;
