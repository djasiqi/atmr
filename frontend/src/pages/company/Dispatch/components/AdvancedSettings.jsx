// frontend/src/pages/company/Dispatch/components/AdvancedSettings.jsx
/**
 * Composant de configuration avancée pour le dispatch.
 * Permet de personnaliser les overrides (heuristic, solver, fairness, etc.)
 */

import React, { useState, useEffect } from 'react';
import PropTypes from 'prop-types';
import './AdvancedSettings.css';

const AdvancedSettings = ({ onApply, initialSettings = {}, drivers = [] }) => {
  // État local pour les overrides (utilise uniquement initialSettings fourni par le parent)
  const [overrides, setOverrides] = useState(initialSettings);
  const [expanded, setExpanded] = useState({});

  // 🆕 Synchroniser avec initialSettings si elles changent (ex: chargement DB)
  useEffect(() => {
    setOverrides(initialSettings);
  }, [initialSettings]);

  // Toggle section expansion
  const toggleSection = (section) => {
    setExpanded((prev) => ({ ...prev, [section]: !prev[section] }));
  };

  // Update override value
  const updateOverride = (category, key, value) => {
    if (category === 'root') {
      // Pour les paramètres de niveau racine (comme allow_emergency)
      setOverrides((prev) => ({
        ...prev,
        [key]: value,
      }));
    } else {
      setOverrides((prev) => ({
        ...prev,
        [category]: {
          ...prev[category],
          [key]: value,
        },
      }));
    }
  };

  // Reset to defaults
  const resetToDefaults = () => {
    if (window.confirm('Réinitialiser tous les paramètres aux valeurs par défaut ?')) {
      setOverrides({});
    }
  };

  // Apply overrides (ne sauvegarde PAS dans localStorage, délègue au parent)
  const handleApply = () => {
    if (onApply) {
      onApply(overrides);
    }
  };

  return (
    <div className="advanced-settings">
      <div className="settings-header">
        <h3>⚙️ Paramètres Avancés</h3>
        <p className="settings-subtitle">
          Personnalisez le comportement du dispatch selon vos besoins spécifiques
        </p>
      </div>

      {/* Section: Heuristique */}
      <div className="settings-section">
        <div className="section-header" onClick={() => toggleSection('heuristic')}>
          <span className="section-title">🎯 Poids Heuristique</span>
          <span className="section-toggle">{expanded.heuristic ? '▼' : '▶'}</span>
        </div>

        {expanded.heuristic && (
          <div className="section-content">
            <p className="section-description">
              Ajustez l'importance relative de chaque critère dans l'algorithme de dispatch
            </p>

            <div className="setting-item">
              <label>Proximité (0-1)</label>
              <input
                type="number"
                min="0"
                max="1"
                step="0.01"
                value={overrides.heuristic?.proximity || 0.2}
                onChange={(e) =>
                  updateOverride('heuristic', 'proximity', parseFloat(e.target.value))
                }
              />
              <span className="setting-help">Distance/temps vers le pickup</span>
            </div>

            <div className="setting-item">
              <label>Équilibre charge (0-1)</label>
              <input
                type="number"
                min="0"
                max="1"
                step="0.01"
                value={overrides.heuristic?.driver_load_balance || 0.7}
                onChange={(e) =>
                  updateOverride('heuristic', 'driver_load_balance', parseFloat(e.target.value))
                }
              />
              <span className="setting-help">Répartition équitable entre chauffeurs</span>
            </div>

            <div className="setting-item">
              <label>Priorité (0-1)</label>
              <input
                type="number"
                min="0"
                max="1"
                step="0.01"
                value={overrides.heuristic?.priority || 0.06}
                onChange={(e) =>
                  updateOverride('heuristic', 'priority', parseFloat(e.target.value))
                }
              />
              <span className="setting-help">Courses médicales ou VIP</span>
            </div>
          </div>
        )}
      </div>

      {/* Section: Solver OR-Tools */}
      <div className="settings-section">
        <div className="section-header" onClick={() => toggleSection('solver')}>
          <span className="section-title">🔧 Optimiseur (OR-Tools)</span>
          <span className="section-toggle">{expanded.solver ? '▼' : '▶'}</span>
        </div>

        {expanded.solver && (
          <div className="section-content">
            <p className="section-description">Paramètres du solveur d'optimisation avancé</p>

            <div className="setting-item">
              <label>Temps limite (secondes)</label>
              <input
                type="number"
                min="10"
                max="300"
                value={overrides.solver?.time_limit_sec || 60}
                onChange={(e) =>
                  updateOverride('solver', 'time_limit_sec', parseInt(e.target.value))
                }
              />
              <span className="setting-help">Temps max pour trouver solution optimale</span>
            </div>

            <div className="setting-item">
              <label>Courses max par chauffeur</label>
              <input
                type="number"
                min="1"
                max="12"
                value={overrides.solver?.max_bookings_per_driver || 6}
                onChange={(e) =>
                  updateOverride('solver', 'max_bookings_per_driver', parseInt(e.target.value))
                }
              />
              <span className="setting-help">Limite de charge par chauffeur</span>
            </div>

            <div className="setting-item">
              <label>Pénalité non-assigné</label>
              <input
                type="number"
                min="1000"
                max="50000"
                step="1000"
                value={overrides.solver?.unassigned_penalty_base || 10000}
                onChange={(e) =>
                  updateOverride('solver', 'unassigned_penalty_base', parseInt(e.target.value))
                }
              />
              <span className="setting-help">Coût de laisser une course non-assignée</span>
            </div>
          </div>
        )}
      </div>

      {/* Section: Temps de service */}
      <div className="settings-section">
        <div className="section-header" onClick={() => toggleSection('service_times')}>
          <span className="section-title">⏱️ Temps de Service</span>
          <span className="section-toggle">{expanded.service_times ? '▼' : '▶'}</span>
        </div>

        {expanded.service_times && (
          <div className="section-content">
            <p className="section-description">Durées moyennes des opérations de service</p>

            <div className="setting-item">
              <label>Pickup (minutes)</label>
              <input
                type="number"
                min="1"
                max="30"
                value={overrides.service_times?.pickup_service_min || 5}
                onChange={(e) =>
                  updateOverride('service_times', 'pickup_service_min', parseInt(e.target.value))
                }
              />
              <span className="setting-help">Temps moyen pour embarquer client</span>
            </div>

            <div className="setting-item">
              <label>Dropoff (minutes)</label>
              <input
                type="number"
                min="1"
                max="30"
                value={overrides.service_times?.dropoff_service_min || 10}
                onChange={(e) =>
                  updateOverride('service_times', 'dropoff_service_min', parseInt(e.target.value))
                }
              />
              <span className="setting-help">Temps moyen pour déposer client</span>
            </div>

            <div className="setting-item">
              <label>Marge transition (minutes)</label>
              <input
                type="number"
                min="5"
                max="60"
                value={overrides.service_times?.min_transition_margin_min || 15}
                onChange={(e) =>
                  updateOverride(
                    'service_times',
                    'min_transition_margin_min',
                    parseInt(e.target.value)
                  )
                }
              />
              <span className="setting-help">Marge minimale entre deux courses</span>
            </div>
          </div>
        )}
      </div>

      {/* Section: Regroupement (Pooling) */}
      <div className="settings-section">
        <div className="section-header" onClick={() => toggleSection('pooling')}>
          <span className="section-title">👥 Regroupement de Courses</span>
          <span className="section-toggle">{expanded.pooling ? '▼' : '▶'}</span>
        </div>

        {expanded.pooling && (
          <div className="section-content">
            <p className="section-description">
              Paramètres pour le regroupement de courses (ride-pooling)
            </p>

            <div className="setting-item">
              <label>
                <input
                  type="checkbox"
                  checked={overrides.pooling?.enabled !== false}
                  onChange={(e) => updateOverride('pooling', 'enabled', e.target.checked)}
                />
                Activer le regroupement
              </label>
              <span className="setting-help">Permet de combiner plusieurs courses compatibles</span>
            </div>

            {overrides.pooling?.enabled !== false && (
              <>
                <div className="setting-item">
                  <label>Tolérance temporelle (minutes)</label>
                  <input
                    type="number"
                    min="5"
                    max="30"
                    value={overrides.pooling?.time_tolerance_min || 10}
                    onChange={(e) =>
                      updateOverride('pooling', 'time_tolerance_min', parseInt(e.target.value))
                    }
                  />
                  <span className="setting-help">Écart maximal entre heures de pickup</span>
                </div>

                <div className="setting-item">
                  <label>Distance pickup max (mètres)</label>
                  <input
                    type="number"
                    min="100"
                    max="2000"
                    step="100"
                    value={overrides.pooling?.pickup_distance_m || 500}
                    onChange={(e) =>
                      updateOverride('pooling', 'pickup_distance_m', parseInt(e.target.value))
                    }
                  />
                  <span className="setting-help">Distance maximale entre lieux de pickup</span>
                </div>

                <div className="setting-item">
                  <label>Détour max (minutes)</label>
                  <input
                    type="number"
                    min="5"
                    max="30"
                    value={overrides.pooling?.max_detour_min || 15}
                    onChange={(e) =>
                      updateOverride('pooling', 'max_detour_min', parseInt(e.target.value))
                    }
                  />
                  <span className="setting-help">Détour maximal acceptable pour dropoffs</span>
                </div>
              </>
            )}
          </div>
        )}
      </div>

      {/* Section: Équité */}
      <div className="settings-section">
        <div className="section-header" onClick={() => toggleSection('fairness')}>
          <span className="section-title">⚖️ Équité Chauffeurs</span>
          <span className="section-toggle">{expanded.fairness ? '▼' : '▶'}</span>
        </div>

        {expanded.fairness && (
          <div className="section-content">
            <p className="section-description">
              Paramètres pour assurer une répartition équitable des courses
            </p>

            <div className="setting-item">
              <label>
                <input
                  type="checkbox"
                  checked={overrides.fairness?.enable_fairness !== false}
                  onChange={(e) => updateOverride('fairness', 'enable_fairness', e.target.checked)}
                />
                Activer l'équité
              </label>
              <span className="setting-help">Équilibre la charge entre chauffeurs</span>
            </div>

            {overrides.fairness?.enable_fairness !== false && (
              <>
                <div className="setting-item">
                  <label>Fenêtre d'équité (jours)</label>
                  <input
                    type="number"
                    min="1"
                    max="30"
                    value={overrides.fairness?.fairness_window_days || 7}
                    onChange={(e) =>
                      updateOverride('fairness', 'fairness_window_days', parseInt(e.target.value))
                    }
                  />
                  <span className="setting-help">Période sur laquelle l'équité est calculée</span>
                </div>

                <div className="setting-item">
                  <label>Poids équité (0-1)</label>
                  <input
                    type="number"
                    min="0"
                    max="1"
                    step="0.1"
                    value={overrides.fairness?.fairness_weight || 0.3}
                    onChange={(e) =>
                      updateOverride('fairness', 'fairness_weight', parseFloat(e.target.value))
                    }
                  />
                  <span className="setting-help">Importance de l'équité dans l'algorithme</span>
                </div>
              </>
            )}
          </div>
        )}
      </div>

      {/* Section: Préférence Chauffeur */}
      <div className="settings-section">
        <div className="section-header" onClick={() => toggleSection('driver_preference')}>
          <span className="section-title">👤 Préférence Chauffeur</span>
          <span className="section-toggle">{expanded.driver_preference ? '▼' : '▶'}</span>
        </div>

        {expanded.driver_preference && (
          <div className="section-content">
            <p className="section-description">
              Définir un chauffeur préféré pour prioriser ses assignments. Si aucun chauffeur n'est
              sélectionné, l'équité stricte sera appliquée (max 2 courses d'écart).
            </p>

            <div className="setting-item">
              <label>Chauffeur préféré</label>
              <select
                value={overrides.preferred_driver_id || ''}
                onChange={(e) =>
                  updateOverride(
                    'root',
                    'preferred_driver_id',
                    e.target.value ? parseInt(e.target.value) : null
                  )
                }
              >
                <option value="">Aucun (équité stricte)</option>
                {drivers
                  .filter((d) => !d.is_emergency) // Seulement les chauffeurs réguliers
                  .map((driver) => (
                    <option key={driver.id} value={driver.id}>
                      {driver.username || driver.full_name || `Chauffeur #${driver.id}`}
                    </option>
                  ))}
              </select>
              <span className="setting-help">
                Si sélectionné, ce chauffeur sera priorisé. Sinon, équité stricte (max 2 courses
                d'écart).
              </span>
            </div>

            {/* Multiplicateur de charge pour le chauffeur préféré */}
            {overrides.preferred_driver_id && (
              <div className="setting-item">
                <label>Multiplicateur de charge (chauffeur préféré)</label>
                <input
                  type="number"
                  min="1.0"
                  max="3.0"
                  step="0.1"
                  value={overrides.driver_load_multipliers?.[overrides.preferred_driver_id] || 1.5}
                  onChange={(e) => {
                    const multipliers = overrides.driver_load_multipliers || {};
                    updateOverride('root', 'driver_load_multipliers', {
                      ...multipliers,
                      [overrides.preferred_driver_id]: parseFloat(e.target.value) || 1.5,
                    });
                  }}
                />
                <span className="setting-help">
                  Permet au chauffeur préféré de prendre plus de courses (1.5 = 50% de plus, 2.0 =
                  100% de plus)
                </span>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Section: Chauffeurs d'urgence */}
      <div className="settings-section">
        <div className="section-header" onClick={() => toggleSection('emergency')}>
          <span className="section-title">🚨 Chauffeurs d'Urgence</span>
          <span className="section-toggle">{expanded.emergency ? '▼' : '▶'}</span>
        </div>

        {expanded.emergency && (
          <div className="section-content">
            <p className="section-description">
              Contrôle l'utilisation des chauffeurs marqués comme "urgence"
            </p>

            <div className="setting-item">
              <label>
                <input
                  type="checkbox"
                  checked={overrides.allow_emergency !== false}
                  onChange={(e) => updateOverride('root', 'allow_emergency', e.target.checked)}
                />
                Autoriser chauffeurs d'urgence
              </label>
              <span className="setting-help">
                Permet d'utiliser les chauffeurs d'urgence si nécessaire
              </span>
            </div>

            {overrides.allow_emergency !== false && (
              <div className="setting-item">
                <label>Pénalité d'utilisation (0-1000)</label>
                <input
                  type="number"
                  min="0"
                  max="1000"
                  step="50"
                  value={overrides.emergency?.emergency_per_stop_penalty || 500}
                  onChange={(e) =>
                    updateOverride(
                      'emergency',
                      'emergency_per_stop_penalty',
                      parseInt(e.target.value)
                    )
                  }
                />
                <span className="setting-help">
                  Plus élevé = chauffeur d'urgence utilisé en dernier recours seulement (Recommandé:
                  500-800)
                </span>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Actions */}
      <div className="settings-actions">
        <button className="btn-reset" onClick={resetToDefaults}>
          🔄 Réinitialiser
        </button>
        <button className="btn-apply" onClick={handleApply}>
          ✅ Appliquer ces paramètres
        </button>
      </div>

      {/* Info helper */}
      <div className="settings-info">
        <p>
          💡 <strong>Note</strong> : Vous pouvez sauvegarder ces paramètres de manière permanente
          dans la page <strong>Paramètres → Opérations</strong>, ou les appliquer temporairement
          pour un dispatch unique.
        </p>
        <p>
          📌 Les paramètres sauvegardés dans "Opérations" seront appliqués automatiquement à tous
          les dispatchs futurs.
        </p>
      </div>
    </div>
  );
};

AdvancedSettings.propTypes = {
  onApply: PropTypes.func.isRequired,
  initialSettings: PropTypes.object,
  drivers: PropTypes.array,
};

export default AdvancedSettings;
