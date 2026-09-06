import React, { useState, useEffect, useCallback } from 'react';
import {
  FiClipboard, FiCpu, FiZap, FiLock,
  FiAlertTriangle, FiCheckCircle, FiClock
} from 'react-icons/fi';
import apiClient from '../utils/apiClient';
import useShadowMode from '../hooks/useShadowMode';
import { getAuthEnv, hasCompanyDispatchSession } from '../utils/webAuthSession';
import './DispatchModeSelector.css';

const canCallCompanyDispatch = () => hasCompanyDispatchSession(getAuthEnv());

/**
 * Composant amélioré de sélection du mode de dispatch autonome.
 * Permet de basculer entre MANUAL, SEMI_AUTO et FULLY_AUTO.
 *
 * Intègre les statuts RL/Shadow Mode pour informer l'utilisateur
 * sur l'état du système d'IA et les recommandations.
 */
const DispatchModeSelector = ({ onModeChange }) => {
  const [currentMode, setCurrentMode] = useState(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);
  const semiAutoLocked = true;
  const fullyAutoLocked = true;

  // Intégration Shadow Mode pour afficher statuts RL
  const {
    isActive: shadowModeActive,
    agreementRate,
    isReadyForPhase2,
    totalComparisons,
    loading: shadowLoading,
  } = useShadowMode({ autoRefresh: false }); // Charger une seule fois

  const fetchCurrentMode = useCallback(async () => {
    if (!canCallCompanyDispatch()) {
      setLoading(false);
      return;
    }
    try {
      const { data } = await apiClient.get('/company_dispatch/mode');
      if (data.dispatch_mode === 'manual' || data.dispatch_mode === 'semi_auto' || data.dispatch_mode === 'fully_auto') {
        setCurrentMode(data.dispatch_mode);
      }
    } catch (err) {
      console.error('Erreur chargement mode:', err);
      setError('Impossible de charger le mode actuel');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchCurrentMode();
  }, [fetchCurrentMode]);

  const handleModeChange = async (newMode) => {
    if (newMode === currentMode) {
      return; // Déjà sur ce mode
    }

    if (newMode === 'semi_auto' && semiAutoLocked) {
      window.alert(
        'Le mode « Semi-Automatique » est actuellement en développement et ne peut pas être activé.'
      );
      return;
    }

    if (newMode === 'fully_auto' && fullyAutoLocked) {
      window.alert(
        'Le mode « Totalement Automatique » est actuellement en développement et ne peut pas être activé.'
      );
      return;
    }

    // Confirmation pour passage en fully_auto
    if (newMode === 'fully_auto') {
      // Vérifier si Shadow Mode validé (Phase 2 prête)
      if (!isReadyForPhase2 && shadowModeActive) {
        const proceedAnyway = window.confirm(
          'ATTENTION : Shadow Mode pas encore validé\n\n' +
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
        'ATTENTION : En mode Totalement Automatique, le système gérera tout automatiquement.\n\n' +
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
      alert(`Mode de dispatch changé : ${modeLabels[newMode]}`);
    } catch (err) {
      console.error('Erreur changement mode:', err);
      const detail =
        err?.response?.data?.error ||
        err?.response?.data?.message ||
        err?.message ||
        'Erreur lors du changement de mode';
      setError(detail);
      alert(detail);
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

  return (
    <div className="dispatch-mode-selector">
      {error && <div className="error-banner"><FiAlertTriangle size={12} /> {error}</div>}

      <div className="mode-cards">
        {/* MODE MANUEL */}
        <div
          className={`mode-card ${currentMode === 'manual' ? 'active' : ''} ${saving ? 'disabled' : ''}`}
          onClick={() => !saving && handleModeChange('manual')}
        >
          <div className="mode-radio"></div>
          <div className="mode-content">
            <div className="mode-title">
              <h3><FiClipboard size={14} /> Manuel</h3>
              <span className="mode-meta">Automatisation 0%</span>
            </div>
            <p className="mode-description">
              Contrôle total sur chaque assignation. Suggestions IA en lecture seule.
            </p>
          </div>
        </div>

        {/* MODE SEMI-AUTO */}
        <div
          className={`mode-card ${currentMode === 'semi_auto' ? 'active' : ''} ${
            saving || semiAutoLocked ? 'disabled' : ''
          }`}
          onClick={() => !saving && !semiAutoLocked && handleModeChange('semi_auto')}
        >
          <div className="mode-radio"></div>
          <div className="mode-content">
            <div className="mode-title">
              <h3><FiCpu size={14} /> Semi-Automatique</h3>
              <span className="mode-badge advanced">
                <FiLock size={10} /> En développement
              </span>
              <span className="mode-meta">Automatisation 50-70%</span>
            </div>
            <p className="mode-description">
              Dispatch optimisé OR-Tools avec suggestions cliquables. Vous validez avant application.
            </p>
            {semiAutoLocked && (
              <div className="mode-notice">
                <FiAlertTriangle size={11} /> Activation bientôt disponible
              </div>
            )}
          </div>
        </div>

        {/* MODE FULLY AUTO */}
        <div
          className={`mode-card ${currentMode === 'fully_auto' ? 'active' : ''} ${
            saving || fullyAutoLocked ? 'disabled' : ''
          }`}
          onClick={() => !saving && !fullyAutoLocked && handleModeChange('fully_auto')}
        >
          <div className="mode-radio"></div>
          <div className="mode-content">
            <div className="mode-title">
              <h3><FiZap size={14} /> Totalement Automatique</h3>
              <span className="mode-badge advanced">
                <FiLock size={10} /> En développement
              </span>
              <span className="mode-meta">Automatisation 90-95%</span>
            </div>
            <p className="mode-description">
              Système 100% autonome. Application automatique des décisions haute confiance.
            </p>
            {fullyAutoLocked && (
              <div className="mode-notice">
                <FiAlertTriangle size={11} /> Activation bientôt disponible
              </div>
            )}
            {!fullyAutoLocked && !isReadyForPhase2 && shadowModeActive && (
              <div className="mode-notice warning">
                <FiClock size={11} /> Shadow Mode en validation ({(agreementRate * 100).toFixed(0)}%)
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

      {/* Info contextuelle — une seule ligne sobre */}
      {isReadyForPhase2 && (
        <div className="mode-hint success">
          <FiCheckCircle size={12} /> MDI validé — passage en Fully Auto possible.
        </div>
      )}
      {!isReadyForPhase2 && shadowModeActive && !shadowLoading && (
        <div className="mode-hint">
          <FiClock size={12} /> Shadow Mode en cours — {(agreementRate * 100).toFixed(0)}% accord, {totalComparisons} comparaisons.
        </div>
      )}
    </div>
  );
};

export default DispatchModeSelector;
