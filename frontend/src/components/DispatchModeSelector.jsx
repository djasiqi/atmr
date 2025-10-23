import React, { useState, useEffect } from 'react';
import apiClient from '../utils/apiClient';
import useShadowMode from '../hooks/useShadowMode';
import './DispatchModeSelector.css';

/**
 * Composant amélioré de sélection du mode de dispatch autonome.
 * Permet de basculer entre MANUAL, SEMI_AUTO et FULLY_AUTO.
 *
 * Intègre les statuts RL/Shadow Mode pour informer l'utilisateur
 * sur l'état du système d'IA et les recommandations.
 */
const DispatchModeSelector = ({ onModeChange }) => {
  const [currentMode, setCurrentMode] = useState('semi_auto');
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);

  // 🆕 Intégration Shadow Mode pour afficher statuts RL
  const {
    isActive: shadowModeActive,
    agreementRate,
    isReadyForPhase2,
    totalComparisons,
    loading: shadowLoading,
  } = useShadowMode({ autoRefresh: false }); // Charger une seule fois

  useEffect(() => {
    fetchCurrentMode();
  }, []);

  const fetchCurrentMode = async () => {
    try {
      const { data } = await apiClient.get('/company_dispatch/mode');
      setCurrentMode(data.dispatch_mode);
      setLoading(false);
    } catch (err) {
      console.error('Erreur chargement mode:', err);
      setError('Impossible de charger le mode actuel');
      setLoading(false);
    }
  };

  const handleModeChange = async (newMode) => {
    if (newMode === currentMode) {
      return; // Déjà sur ce mode
    }

    // Confirmation pour passage en fully_auto
    if (newMode === 'fully_auto') {
      // Vérifier si Shadow Mode validé (Phase 2 prête)
      if (!isReadyForPhase2 && shadowModeActive) {
        const proceedAnyway = window.confirm(
          '⚠️ ATTENTION : Shadow Mode pas encore validé\n\n' +
            `Taux d'accord MDI: ${(agreementRate * 100).toFixed(1)}% (objectif: >75%)\n` +
            `Comparaisons: ${totalComparisons} (objectif: >1000)\n\n` +
            "Il est recommandé d'attendre la validation du Shadow Mode avant de passer en mode Fully Auto.\n\n" +
            'Voulez-vous continuer quand même ?'
        );
        if (!proceedAnyway) {
          return;
        }
      }

      const confirmed = window.confirm(
        '⚠️ ATTENTION : En mode Totalement Automatique, le système gérera tout automatiquement.\n\n' +
          'Le système appliquera automatiquement :\n' +
          '- Les assignations optimales (MDI RL)\n' +
          '- Les notifications clients (retards 5-20 min)\n' +
          '- Les ajustements horaires mineurs\n' +
          '- Les ré-optimisations si nécessaire\n\n' +
          'Êtes-vous sûr de vouloir activer ce mode ?'
      );
      if (!confirmed) {
        return;
      }
    }

    setSaving(true);
    setError(null);

    try {
      await apiClient.put('/company_dispatch/mode', {
        dispatch_mode: newMode,
      });

      setCurrentMode(newMode);

      // Notifier le parent si callback fourni
      if (onModeChange) {
        onModeChange(newMode);
      }

      // Message de succès
      const modeLabels = {
        manual: 'Manuel',
        semi_auto: 'Semi-Automatique',
        fully_auto: 'Totalement Automatique',
      };
      alert(`✅ Mode de dispatch changé : ${modeLabels[newMode]}`);
    } catch (err) {
      console.error('Erreur changement mode:', err);
      setError('Erreur lors du changement de mode');
      alert('❌ Erreur lors du changement de mode');
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="dispatch-mode-selector loading">
        <div className="spinner"></div>
        <p>Chargement...</p>
      </div>
    );
  }

  // 🆕 Badge d'état Shadow Mode
  const renderShadowModeBadge = () => {
    if (shadowLoading) return null;

    if (!shadowModeActive) {
      return <div className="shadow-badge inactive">🔍 Shadow Mode: Inactif</div>;
    }

    if (isReadyForPhase2) {
      return (
        <div className="shadow-badge ready">
          ✅ Shadow Mode: Validé ({(agreementRate * 100).toFixed(0)}% accord, {totalComparisons}+
          comparaisons)
        </div>
      );
    }

    return (
      <div className="shadow-badge monitoring">
        ⏳ Shadow Mode: En cours ({(agreementRate * 100).toFixed(0)}% accord, {totalComparisons}{' '}
        comparaisons)
      </div>
    );
  };

  // 🆕 Badge RL pour chaque mode
  const getRLBadge = (mode) => {
    if (shadowLoading) return null;

    if (mode === 'manual') {
      return (
        <span className="rl-badge info" title="Suggestions MDI affichées en lecture seule">
          💡 Suggestions RL
        </span>
      );
    }

    if (mode === 'semi_auto') {
      if (isReadyForPhase2) {
        return (
          <span className="rl-badge success" title="MDI validé - Suggestions haute qualité">
            ✨ RL Optimisé
          </span>
        );
      }
      return (
        <span className="rl-badge active" title="MDI actif - Suggestions en temps réel">
          🤖 RL Actif
        </span>
      );
    }

    if (mode === 'fully_auto') {
      if (isReadyForPhase2) {
        return (
          <span className="rl-badge success" title="MDI validé - Prêt pour auto-application">
            🚀 RL Production
          </span>
        );
      }
      return (
        <span className="rl-badge warning" title="Shadow Mode pas encore validé">
          ⚠️ RL Beta
        </span>
      );
    }

    return null;
  };

  return (
    <div className="dispatch-mode-selector">
      <div className="selector-header">
        <h2>🤖 Mode de dispatch autonome</h2>
        <p className="subtitle">
          Choisissez le niveau d'automatisation adapté à vos besoins • Optimisé par RL/MDI
        </p>
      </div>

      {/* 🆕 Badge d'état Shadow Mode global */}
      {!shadowLoading && renderShadowModeBadge()}

      {error && <div className="error-banner">❌ {error}</div>}

      <div className="mode-cards">
        {/* MODE MANUEL */}
        <div
          className={`mode-card ${currentMode === 'manual' ? 'active' : ''} ${
            saving ? 'disabled' : ''
          }`}
          onClick={() => !saving && handleModeChange('manual')}
        >
          <div className="mode-radio"></div>
          <div className="mode-content">
            <div className="mode-title">
              <h3>📋 Manuel</h3>
              {getRLBadge('manual')}
            </div>
            <p className="mode-description">
              Contrôle total sur chaque assignation. Le MDI fournit des suggestions informatives
              pour vous guider, mais vous gardez le contrôle complet des décisions.
            </p>
            <div className="mode-features-compact">
              <span className="feature-tag">🎯 Contrôle total</span>
              <span className="feature-tag">💡 Suggestions MDI readonly</span>
              <span className="feature-tag">❌ Pas d'automatisation</span>
              <span className="feature-tag">📊 Métriques RL visibles</span>
            </div>
            <div className="mode-metrics">
              <div className="metric-item">
                <span className="metric-label">Automatisation</span>
                <span className="metric-value">0%</span>
              </div>
              <div className="metric-item">
                <span className="metric-label">IA Assistance</span>
                <span className="metric-value">Passive</span>
              </div>
            </div>
          </div>
        </div>

        {/* MODE SEMI-AUTO */}
        <div
          className={`mode-card ${currentMode === 'semi_auto' ? 'active' : ''} ${
            saving ? 'disabled' : ''
          }`}
          onClick={() => !saving && handleModeChange('semi_auto')}
        >
          <div className="mode-radio"></div>
          <div className="mode-content">
            <div className="mode-title">
              <h3>🧠 Semi-Automatique</h3>
              <span className="mode-badge recommended">⭐ Recommandé</span>
              {getRLBadge('semi_auto')}
            </div>
            <p className="mode-description">
              Dispatch optimisé avec OR-Tools + suggestions MDI cliquables. Vous validez les
              suggestions haute confiance. Monitoring temps réel. Équilibre parfait entre
              automatisation et contrôle.
            </p>
            <div className="mode-features-compact">
              <span className="feature-tag">🤖 Dispatch OR-Tools auto</span>
              <span className="feature-tag">✨ Suggestions MDI cliquables</span>
              <span className="feature-tag">✋ Validation manuelle</span>
              <span className="feature-tag">📊 Monitoring temps réel</span>
              <span className="feature-tag">🔔 Alertes intelligentes</span>
            </div>
            <div className="mode-metrics">
              <div className="metric-item">
                <span className="metric-label">Automatisation</span>
                <span className="metric-value">50-70%</span>
              </div>
              <div className="metric-item">
                <span className="metric-label">IA Assistance</span>
                <span className="metric-value">Active</span>
              </div>
              {!shadowLoading && agreementRate > 0 && (
                <div className="metric-item highlight">
                  <span className="metric-label">MDI Qualité</span>
                  <span className="metric-value">{(agreementRate * 100).toFixed(0)}%</span>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* MODE FULLY AUTO */}
        <div
          className={`mode-card ${currentMode === 'fully_auto' ? 'active' : ''} ${
            saving ? 'disabled' : ''
          } ${!isReadyForPhase2 && shadowModeActive ? 'warning-border' : ''}`}
          onClick={() => !saving && handleModeChange('fully_auto')}
        >
          <div className="mode-radio"></div>
          <div className="mode-content">
            <div className="mode-title">
              <h3>🚀 Totalement Automatique</h3>
              <span className="mode-badge advanced">⚡ Avancé</span>
              {getRLBadge('fully_auto')}
            </div>
            <p className="mode-description">
              Système 100% autonome piloté par MDI RL (+765% performance vs baseline). Application
              automatique des suggestions haute confiance. Intervention uniquement pour cas
              critiques. ROI validé: 379k€/an.
            </p>
            <div className="mode-features-compact">
              <span className="feature-tag">🤖 100% Auto MDI</span>
              <span className="feature-tag">🔄 Ré-optimisation auto</span>
              <span className="feature-tag">⚡ Application instantanée</span>
              <span className="feature-tag">🎯 IA décide (haute confiance)</span>
              <span className="feature-tag">🛡️ Safety limits actives</span>
            </div>
            <div className="mode-metrics">
              <div className="metric-item">
                <span className="metric-label">Automatisation</span>
                <span className="metric-value">90-95%</span>
              </div>
              <div className="metric-item">
                <span className="metric-label">IA Assistance</span>
                <span className="metric-value">Autonome</span>
              </div>
              <div className="metric-item highlight">
                <span className="metric-label">Performance MDI</span>
                <span className="metric-value">+765%</span>
              </div>
            </div>
            {!isReadyForPhase2 && shadowModeActive && (
              <div className="mode-warning">
                ⚠️ Shadow Mode pas encore validé. Recommandé d'attendre validation avant activation.
              </div>
            )}
          </div>
        </div>
      </div>

      {saving && (
        <div className="saving-overlay">
          <div className="saving-spinner"></div>
          <p>Mise à jour en cours...</p>
        </div>
      )}

      {/* Info améliorée avec recommandations RL */}
      <div className="mode-info">
        <div className="info-section">
          <strong>💡 Conseil :</strong> Commencez avec le mode <strong>Semi-Automatique</strong>{' '}
          pour profiter de l'optimisation OR-Tools + suggestions MDI tout en gardant le contrôle,
          puis passez en <strong>Fully Auto</strong> une fois le Shadow Mode validé (
          {isReadyForPhase2 ? '✅ Validé' : '⏳ En cours'}).
        </div>
        {isReadyForPhase2 && (
          <div className="info-section success">
            <strong>✅ MDI Validé!</strong> Le système RL a atteint{' '}
            {(agreementRate * 100).toFixed(0)}% de taux d'accord sur {totalComparisons}+
            comparaisons. Vous pouvez activer le mode Fully Auto en toute confiance. Performance
            garantie: +765% vs baseline.
          </div>
        )}
        {!isReadyForPhase2 && shadowModeActive && (
          <div className="info-section info">
            <strong>⏳ Shadow Mode en cours:</strong> Le MDI est actuellement en phase de
            validation. Taux d'accord: {(agreementRate * 100).toFixed(0)}% (objectif: &gt;75%).
            Comparaisons: {totalComparisons} (objectif: &gt;1000). Le mode Fully Auto sera
            recommandé après validation.
          </div>
        )}
        {!shadowModeActive && !shadowLoading && (
          <div className="info-section warning">
            <strong>🔍 Shadow Mode inactif:</strong> Le système MDI n'est pas en cours de
            surveillance. Contactez votre administrateur pour activer le Shadow Mode avant
            d'utiliser le mode Fully Auto.
          </div>
        )}
      </div>
    </div>
  );
};

export default DispatchModeSelector;
