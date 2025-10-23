import React, { useState, useEffect } from 'react';
import apiClient from '../utils/apiClient';
import './AutonomousConfigPanel.css';

/**
 * Panneau de configuration avancée pour le mode fully_auto.
 * Design sobre et cohérent avec le reste du site.
 */
const AutonomousConfigPanel = ({ visible = true }) => {
  const [config, setConfig] = useState(null);
  const [mode, setMode] = useState('semi_auto');
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (visible) {
      fetchConfig();
    }
  }, [visible]);

  const fetchConfig = async () => {
    try {
      const { data } = await apiClient.get('/company_dispatch/mode');
      setMode(data.dispatch_mode);
      setConfig(data.autonomous_config);
      setLoading(false);
    } catch (err) {
      console.error('Erreur chargement config:', err);
      setLoading(false);
    }
  };

  const handleConfigChange = async (section, key, value) => {
    const newConfig = {
      ...config,
      [section]: {
        ...config[section],
        [key]: value,
      },
    };

    setSaving(true);

    try {
      await apiClient.put('/company_dispatch/mode', {
        autonomous_config: newConfig,
      });

      setConfig(newConfig);
    } catch (err) {
      console.error('Erreur mise à jour config:', err);
      alert('❌ Erreur lors de la mise à jour');
    } finally {
      setSaving(false);
    }
  };

  const handleToggleWithConfirm = (section, key, checked) => {
    if (key === 'reassignments' && checked) {
      const confirmed = window.confirm(
        '⚠️ ATTENTION : Réassignations automatiques\n\n' +
          'Le système pourra changer le chauffeur assigné automatiquement.\n' +
          'Ceci peut créer de la confusion.\n\n' +
          'Confirmer ?'
      );
      if (!confirmed) return;
    }
    handleConfigChange(section, key, checked);
  };

  if (!visible || mode !== 'fully_auto') {
    return null;
  }

  if (loading || !config) {
    return (
      <div className="autonomous-config-panel loading">
        <div className="spinner"></div>
        <p>Chargement...</p>
      </div>
    );
  }

  const autoApplyRules = config.auto_apply_rules || {};
  const safetyLimits = config.safety_limits || {};
  const reOptimizeTriggers = config.re_optimize_triggers || {};

  return (
    <div className="autonomous-config-panel">
      <div className="panel-header">
        <h3>⚡ Configuration avancée</h3>
        <p>Paramètres du mode totalement automatique</p>
      </div>

      {/* Section 1 : Actions automatiques */}
      <div className="config-section">
        <h4>🤖 Actions automatiques</h4>
        <p className="section-description">Ce que le système peut faire sans validation</p>

        <div className="config-options">
          <label className="config-option">
            <div className="option-info">
              <div className="toggle-switch">
                <input
                  type="checkbox"
                  checked={autoApplyRules.customer_notifications ?? true}
                  onChange={(e) =>
                    handleConfigChange(
                      'auto_apply_rules',
                      'customer_notifications',
                      e.target.checked
                    )
                  }
                  disabled={saving}
                />
                <span className="toggle-slider"></span>
              </div>
              <div className="option-text">
                <strong>Notifications clients</strong>
                <span className="option-detail">Alertes auto si retard 5-20 min</span>
              </div>
            </div>
            <span className="recommended-tag">✓ Recommandé</span>
          </label>

          <label className="config-option">
            <div className="option-info">
              <div className="toggle-switch">
                <input
                  type="checkbox"
                  checked={autoApplyRules.minor_time_adjustments ?? false}
                  onChange={(e) =>
                    handleConfigChange(
                      'auto_apply_rules',
                      'minor_time_adjustments',
                      e.target.checked
                    )
                  }
                  disabled={saving}
                />
                <span className="toggle-slider"></span>
              </div>
              <div className="option-text">
                <strong>Ajustements horaires</strong>
                <span className="option-detail">Ajuster auto si retard &lt; 10 min</span>
              </div>
            </div>
          </label>

          <label className="config-option warning">
            <div className="option-info">
              <div className="toggle-switch">
                <input
                  type="checkbox"
                  checked={autoApplyRules.reassignments ?? false}
                  onChange={(e) =>
                    handleToggleWithConfirm('auto_apply_rules', 'reassignments', e.target.checked)
                  }
                  disabled={saving}
                />
                <span className="toggle-slider"></span>
              </div>
              <div className="option-text">
                <strong>Réassignations auto</strong>
                <span className="option-detail">Changer chauffeur si meilleur dispo</span>
              </div>
            </div>
            <span className="risky-tag">⚠️ Risqué</span>
          </label>

          <label className="config-option">
            <div className="option-info">
              <div className="toggle-switch">
                <input
                  type="checkbox"
                  checked={autoApplyRules.emergency_notifications ?? true}
                  onChange={(e) =>
                    handleConfigChange(
                      'auto_apply_rules',
                      'emergency_notifications',
                      e.target.checked
                    )
                  }
                  disabled={saving}
                />
                <span className="toggle-slider"></span>
              </div>
              <div className="option-text">
                <strong>Alertes urgence</strong>
                <span className="option-detail">Notification si retard &gt; 30 min</span>
              </div>
            </div>
            <span className="recommended-tag">✓ Recommandé</span>
          </label>
        </div>
      </div>

      {/* Section 2 : Limites de sécurité */}
      <div className="config-section">
        <h4>🛡️ Limites de sécurité</h4>
        <p className="section-description">Protections contre actions excessives</p>

        <div className="number-input-group">
          <div className="input-field">
            <label>Actions max/heure</label>
            <input
              type="number"
              min="1"
              max="200"
              value={safetyLimits.max_auto_actions_per_hour ?? 50}
              onChange={(e) =>
                handleConfigChange(
                  'safety_limits',
                  'max_auto_actions_per_hour',
                  parseInt(e.target.value)
                )
              }
              disabled={saving}
            />
            <span className="hint">Limite globale d'actions</span>
          </div>

          <div className="input-field">
            <label>Réassignations max/jour</label>
            <input
              type="number"
              min="0"
              max="50"
              value={safetyLimits.max_auto_reassignments_per_day ?? 10}
              onChange={(e) =>
                handleConfigChange(
                  'safety_limits',
                  'max_auto_reassignments_per_day',
                  parseInt(e.target.value)
                )
              }
              disabled={saving}
            />
            <span className="hint">Protection réassignations</span>
          </div>
        </div>

        <div className="number-input-group">
          <div className="input-field">
            <label>Seuil validation (min)</label>
            <input
              type="number"
              min="10"
              max="120"
              value={safetyLimits.require_approval_delay_minutes ?? 30}
              onChange={(e) =>
                handleConfigChange(
                  'safety_limits',
                  'require_approval_delay_minutes',
                  parseInt(e.target.value)
                )
              }
              disabled={saving}
            />
            <span className="hint">Retards &gt; X → validation requise</span>
          </div>
        </div>
      </div>

      {/* Section 3 : Déclencheurs ré-optimisation */}
      <div className="config-section">
        <h4>🔄 Ré-optimisation auto</h4>
        <p className="section-description">Quand relancer un dispatch automatiquement</p>

        <div className="number-input-group">
          <div className="input-field">
            <label>Seuil retard (min)</label>
            <input
              type="number"
              min="5"
              max="60"
              value={reOptimizeTriggers.delay_threshold_minutes ?? 15}
              onChange={(e) =>
                handleConfigChange(
                  're_optimize_triggers',
                  'delay_threshold_minutes',
                  parseInt(e.target.value)
                )
              }
              disabled={saving}
            />
            <span className="hint">Re-dispatch si retard &gt; X min</span>
          </div>

          <div className="input-field">
            <label>Gain minimal (min)</label>
            <input
              type="number"
              min="5"
              max="30"
              value={reOptimizeTriggers.better_driver_available_gain_minutes ?? 10}
              onChange={(e) =>
                handleConfigChange(
                  're_optimize_triggers',
                  'better_driver_available_gain_minutes',
                  parseInt(e.target.value)
                )
              }
              disabled={saving}
            />
            <span className="hint">Re-dispatch si gain &gt; X min</span>
          </div>
        </div>

        <div className="config-options">
          <label className="config-option">
            <div className="option-info">
              <div className="toggle-switch">
                <input
                  type="checkbox"
                  checked={reOptimizeTriggers.driver_became_unavailable ?? true}
                  onChange={(e) =>
                    handleConfigChange(
                      're_optimize_triggers',
                      'driver_became_unavailable',
                      e.target.checked
                    )
                  }
                  disabled={saving}
                />
                <span className="toggle-slider"></span>
              </div>
              <div className="option-text">
                <strong>Si chauffeur indisponible</strong>
                <span className="option-detail">Re-dispatch auto si assigné devient indispo</span>
              </div>
            </div>
          </label>
        </div>
      </div>

      {/* Info box */}
      <div className="info-box">
        <h5>ℹ️ À savoir</h5>
        <ul>
          <li>
            Paramètres actifs uniquement en mode <strong>Totalement Automatique</strong>
          </li>
          <li>Limites de sécurité toujours respectées</li>
          <li>Actions critiques → validation manuelle obligatoire</li>
        </ul>
      </div>
    </div>
  );
};

export default AutonomousConfigPanel;
